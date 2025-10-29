import numpy as np
import sys
import os
# add med sam path
import sys
sam3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SAM-Med3D"))
sys.path.append(sam3d_path)
from medim_infer_cp import build_cp_model

import torch
import napari
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QComboBox
from PyQt5 import QtGui  
from medim_infer_cp import create_gt_arr
from medim_infer_cp import sam_model_infer
from medim_infer_cp import data_postprocess
from medim_infer_cp import data_preprocess
from medim_infer_cp import read_data
import os.path as osp

class SAMQWidget3D(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.points_layer = None
        self._last_points = []
        self._segemented_points = []
        self._largest_instance = 0
        self._commited_layers = None
        self._current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded_sam_model = None


        sam_ckpt = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "checkpoints", "sam3D", "sam_model_dice_best.pth")
        )
        self._default_ckpt = sam_ckpt
        self.layout = QtWidgets.QVBoxLayout()

        # 1. log 
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        
        # 2. Input Settings Panel 
        input_group = QtWidgets.QGroupBox("Input Settings")
        input_layout = QtWidgets.QHBoxLayout()

        # label + dropdown + refresh
        self.input_image_label = QtWidgets.QLabel("Select Input Image:")
        self.input_image_dropdown = QtWidgets.QComboBox()
        self.refresh_input_button = QtWidgets.QPushButton("Refresh")
        self.refresh_input_button.clicked.connect(self.update_input_images)

        input_layout.addWidget(self.input_image_label)
        input_layout.addWidget(self.input_image_dropdown)
        input_layout.addWidget(self.refresh_input_button)
        input_group.setLayout(input_layout)

        self.layout.addWidget(input_group)
        self.update_input_images()

        # 3. Load model path and also set model device
        model_group = QtWidgets.QGroupBox("Model Settings")
        model_group_layout = QtWidgets.QVBoxLayout()

        # Model path row
        path_layout = QtWidgets.QHBoxLayout()
        self.model_label = QtWidgets.QLabel("Model Path (.pth):")
        self.model_ckpt = QtWidgets.QLineEdit(self._default_ckpt)
        self.model_browse_button = QtWidgets.QPushButton("Browse")
        self.model_browse_button.clicked.connect(self.browse_model_file)
        path_layout.addWidget(self.model_label)
        path_layout.addWidget(self.model_ckpt)
        path_layout.addWidget(self.model_browse_button)

        # Device row
        device_layout = QtWidgets.QHBoxLayout()
        self.model_device_label = QtWidgets.QLabel("Model Device:")
        self.device_combo = QtWidgets.QComboBox()
        self.detect_devices()

        index = self.device_combo.findText(self._current_device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)

        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_layout.addWidget(self.model_device_label)
        device_layout.addWidget(self.device_combo)

        # Load button
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.clicked.connect(self.load_model)
        load_layout = QtWidgets.QHBoxLayout()
        load_layout.addStretch()
        load_layout.addWidget(self.load_button)

        # Assemble panel
        model_group_layout.addLayout(path_layout)
        model_group_layout.addLayout(device_layout)
        model_group_layout.addLayout(load_layout)
        model_group.setLayout(model_group_layout)

        # Add to main layout
        self.layout.addWidget(model_group)

        # 3. create a points layer
        # about this points layer will it changed according to the selected image layer?
        self.points_layer = self._viewer.add_points(
            np.empty((0, 3)),  # set as 3D points
            name='click area center for segmentation',
            size=3
        )
        self.points_layer.events.data.connect(self._on_points_changed)
        self.log_output.append("add points to click area center for prompt segmentation")
       

        # 4. Saving area
        results_group = QtWidgets.QGroupBox("Results Settings")
        results_layout = QtWidgets.QVBoxLayout()

        output_dir_layout = QtWidgets.QHBoxLayout()
        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_dir)

        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)

      
        hlayout = QtWidgets.QHBoxLayout()
        self.layer_select_label = QtWidgets.QLabel("Select Result Layer:")
        self.layer_combo = QtWidgets.QComboBox()
        self.refresh_layers_button = QtWidgets.QPushButton("Refresh")
        self.refresh_layers_button.clicked.connect(self.update_result_layers)

        self.commit_layers_button = QtWidgets.QPushButton("Commit")
        self.commit_layers_button.clicked.connect(self.commit_selected_prediction)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.save_commited_results)

        hlayout.addWidget(self.layer_select_label)
        hlayout.addWidget(self.layer_combo)
        hlayout.addWidget(self.refresh_layers_button)
        hlayout.addWidget(self.commit_layers_button)
        hlayout.addWidget(self.save_button)
        results_layout.addLayout(output_dir_layout)
        results_layout.addLayout(hlayout)
        results_group.setLayout(results_layout)

      
        self.layout.addWidget(results_group)

        # 5. Run inference area
        self.run_button = QtWidgets.QPushButton("Predict")
        self.run_button.clicked.connect(self.run_inference)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.save_button)

        self.layout.addWidget(self.log_output) # add log at the end
        self.setLayout(self.layout)
        


    def update_input_images(self):
        
        if self._viewer is None:
            print("No viewer attached.")
            return

        self.input_image_dropdown.clear()

        input_layers = [
            layer.name for layer in self._viewer.layers
            if layer.__class__.__name__ == "Image"
        ]

        if not input_layers:
            self.input_image_dropdown.addItem("No image layer found")
            self.input_image_dropdown.setEnabled(False)
        else:
            # [TODO] suppose the image channel is the first one. detect channel automatically?
            self.input_image_dropdown.addItems(input_layers)
            self.input_image_dropdown.setEnabled(True)

    def browse_model_file(self):
        """Open a file selection dialog and update the model file field."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",                    
            "",                                      
            "Model Files (*.pth *.pt *.onnx);;All Files (*)"  
        )
        if file_path:
            self.model_ckpt.setText(file_path)


    def detect_devices(self):
        """Detect available devices and populate the device combo box."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({name})")
        # Optional: Detect MPS / ROCm
        # elif torch.backends.mps.is_available():
        #     devices.append("mps (Apple Silicon)")
        # elif torch.version.hip is not None:
        #     devices.append("rocm")

        self.device_combo.addItems(devices)

    def on_device_changed(self, device_name):
        self._current_device = device_name.split()[0]


    def load_model(self):
        ckpt_path =  self.model_ckpt.text()
        device = torch.device(self._current_device)
        sam_model = build_cp_model(checkpoint=ckpt_path, device=device)
        self._loaded_sam_model = sam_model
        self.log_output.append(f"Model loaded from: {ckpt_path} on device {device}")

    def _on_points_changed(self, event):
        points = self.points_layer.data

        # get new points in the layer
        new_points = [tuple(p) for p in points]
        old_set = set(self._last_points)
        new_set = set(new_points)

        added = new_set - old_set
        removed = old_set - new_set

        if added and not removed:
            for p in added:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"add point: ({coords_str})")

        if removed and not added:
            for p in removed:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"remove point: ({coords_str})")
        
        if added and removed:
            for p in removed:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"move point from: ({coords_str})")
            for p in added:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"move point to: ({coords_str})")

        # update last points
        self._last_points = new_points
       
        # scroll to the end
        self.log_output.moveCursor(QtGui.QTextCursor.End)


    def run_inference(self):
        """Collect UI parameters and execute the backend inference routine."""
        points = self.points_layer.data

        # get new points in the layer
        cur_points = [tuple(p) for p in points]

        old_set = set(self._segemented_points)
        new_set = set(cur_points)
        added = new_set - old_set

        if not added:
            print("No new points added. Please add new points in the viewer.")
            return
        
        else:
            # update segmented points
            self.log_output.append(f"add new points: {added}")
    
        selected_input_name = self.input_image_dropdown.currentText()
        if selected_input_name == "No image layer found":
            QtWidgets.QMessageBox.warning(self, "Warning", "No input image selected.")
            return

        input_layer = self._viewer.layers[selected_input_name]
        img = input_layer.data  

        # Run the backend inference routine
        points_list = []
        clicks = []
        for point in added:
            z0, y0, x0 = point
            points_list.append([z0, y0, x0])

        for item in points_list:
            print('item', item)
            points_dict = {
                'fg': np.array([item]),  # foreground point
                'bg': np.array([], dtype=np.int32).reshape(0, 3)  # background points
            }
            clicks.append(points_dict)

    ## CLICK MUST BE LIKE THIS FORMAT
    #     clicks = [
    #     {
    #         'fg': np.array([[49.        , 243.25277685, 232.56888284]]),
    #         'bg': [(5, 10, 15)]
    #     },
    #     {
    #         'fg': np.array([[49.        , 281.21176995, 147.71936886]]),
    #         'bg': [(35, 45, 55), (30, 40, 50)]
    #     }
    # ]
       
        sam_model = self._loaded_sam_model
        out_dir = self.output_dir_input.text()
        img, spacing, all_clicks, prev_pred = read_data(img, clicks)
        spacing = [1, 1, 1] 
        final_pred = np.zeros_like(img, dtype=np.uint8)
        for idx, cls_clicks in all_clicks.items():
                category_index = idx + 1 + self._largest_instance
                pred_ori = prev_pred==category_index
                final_pred[pred_ori!=0] = category_index
               
                if (cls_clicks[0][1][0] == 1):
                    cls_gt = create_gt_arr(img.shape, cls_clicks[0][0], category_index=category_index)
                    
                    # continue
                    cls_prev_seg = prev_pred==category_index
                    roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(img, cls_gt, cls_prev_seg,
                                                                    orig_spacing=spacing, 
                                                                    category_index=category_index)

                    ''' 3. infer with the pre-trained SAM-Med3D model '''
                    roi_pred = sam_model_infer(sam_model, roi_image, roi_gt=roi_label, prev_low_res_mask=roi_prev_seg)#[Change] change roi_gt from roi_label to None

                    ''' 4. post-process and save the result '''
                    pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
                    final_pred[pred_ori!=0] = category_index
        
        self._largest_instance = np.max(final_pred)

        # output_path = osp.join(out_dir,'test.npy')
        # np.save(output_path, final_pred)
        # print("result saved to", output_path)
        # show results
        self._viewer.add_labels(final_pred, name='SAM Prediction', blending ='additive')

         # update segmented points
        self._segemented_points +=  [tuple(p) for p in added]
        self.log_output.append(f"current segmented points is: {self._segemented_points}")
    
    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)
    
    def update_result_layers(self):
       
        if self._viewer is None:
            print("No viewer attached.")
            return

        self.layer_combo.clear()

        result_layers = [layer.name for layer in self._viewer.layers if 'prediction' in layer.name.lower()]

        if not result_layers:
            self.layer_combo.addItem("No result layer found")
            self.layer_combo.setEnabled(False)
        else:
            self.layer_combo.addItems(result_layers)
            self.layer_combo.setEnabled(True)

    def commit_selected_prediction(self):
        selected_layer_name = self.layer_combo.currentText()
        if not selected_layer_name or selected_layer_name == "No result layer found":
            QtWidgets.QMessageBox.warning(self, "Warning", "No result layer selected.")
            return

        layer = self._viewer.layers[selected_layer_name]
        layer_data = layer.data.copy()

        if self._commited_layers is not None:
            mask = layer_data > 0
            self._commited_layers.data[mask] = layer_data[mask]
        else:
            self._commited_layers = self._viewer.add_labels(
                np.zeros_like(layer_data, dtype=np.uint16), name='Commited'
            )
            self._commited_layers.data = layer_data.copy()

        QtWidgets.QMessageBox.information(self, "Committed", f"Committed layer: {selected_layer_name}")

    def save_commited_results(self):
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return

        save_path = os.path.join(output_dir, f"commited.npy")
        np.save(save_path, self._commited_layers.data)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {save_path}")

