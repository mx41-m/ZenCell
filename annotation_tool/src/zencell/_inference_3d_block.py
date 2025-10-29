import os
import napari
import numpy as np
import torch
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget, QComboBox 
from zencell.zencell_model import models_vit
from zencell.zencell_model.cellpose.dynamics import compute_masks

class InferQWidget3DBlock(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded_model = None

        zencell_ckpt = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "checkpoints", "zencell3D", "checkpoint-4999.pth")
        )

        self._default_ckpt = zencell_ckpt
        self.layout = QtWidgets.QVBoxLayout()

        # 1. Add log panel 
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


        # 4. Run inference area
        self.run_button = QtWidgets.QPushButton("Predict")
        self.run_button.clicked.connect(self.run_inference_block)
        self.layout.addWidget(self.run_button)


        # 5. Saving area
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
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.save_selected_prediction)

        hlayout.addWidget(self.layer_select_label)
        hlayout.addWidget(self.layer_combo)
        hlayout.addWidget(self.refresh_layers_button)
        hlayout.addWidget(self.save_button)
        results_layout.addLayout(output_dir_layout)
        results_layout.addLayout(hlayout)
        results_group.setLayout(results_layout)

      
        self.layout.addWidget(results_group)

        # Initialize once
        self.update_result_layers()

        self.layout.addWidget(self.log_output) # add log at the end
        self.setLayout(self.layout)
        

            
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
        ckpt = self.model_ckpt.text()
        model_name = "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale"
        try:
            device = torch.device(self._current_device)

            model = models_vit.__dict__[model_name](out_chans=4)

            state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(state_dict)
            model = model.half().to(device).eval()
            
            self._loaded_model = model

            self.log_output.append(f"Model loaded successfully on {device}")
            self._selected_device = device

        except Exception as e:
            print("Error loading model:", e)
            self.log_output.append(f"Error loading model: {e}")
            return


    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)


    def run_inference_block(self):
        """Collect UI parameters and execute the backend inference routine."""
        # Gather inputs from UI
        selected_input_name = self.input_image_dropdown.currentText()
        if selected_input_name == "No image layer found":
            QtWidgets.QMessageBox.warning(self, "Warning", "No input image selected.")
            return

        input_layer = self._viewer.layers[selected_input_name]
        input_data = input_layer.data  

        try:
            model = self._loaded_model
        except Exception as e:
            print("Error loading model:", e)
            return

        self.log_output.append("Starting inference...")

        device = self._selected_device
         # Convert numpy array to tensor if needed
        if isinstance(input_data, np.ndarray):
            ref_slice = input_data[0]
            sig_slice = input_data[1]
            ref_tensor = (
                torch.from_numpy(ref_slice)
                if isinstance(ref_slice, np.ndarray)
                else ref_slice
            )
            sig_tensor = (
                torch.from_numpy(sig_slice)
                if isinstance(sig_slice, np.ndarray)
                else sig_slice
            )
            input_vol = torch.stack([ref_tensor, sig_tensor], dim=0).float()

            input_vol = input_vol - input_vol.mean(dim=(1, 2, 3), keepdim=True)
            input_vol = input_vol / (
                input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
            input_vol = input_vol.half()
            input_tensor = input_vol.to(device)

        else:
            input_tensor = input_data.half().to(device)

        # Ensure batch dimension
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)[0]

        # Convert output to numpy
       
        cell_prob = output[0].sigmoid().cpu().float().numpy()
        cell_flow = output[1:].cpu().float().numpy()

        cellmask = compute_masks(
                cell_flow,
                cell_prob,
                min_size=0,
                flow_threshold=None,
                cellprob_threshold=0.5,
                do_3D=True,
            )[0]

        self.log_output.append(f"Inference done, found {len(np.unique(cellmask))-1} unique cells.")
        seg_res = np.zeros((40, 1280, 1280), dtype=np.uint8)
        seg_res[:, 640-128:640+128, 640-128:640+128] = cellmask
        self._viewer.add_labels(
            seg_res, name=f"Zencell Prediction"
        )

        ## add a ROI box
        roi_box = []
        for z in range(0, 40):
            rect_seg = [
                        [z, 640-128, 640-128],  # top-left (z, y, x)
                        [z, 640-128, 640+128],  # top-right
                        [z, 640+128, 640+128],  # bottom-right
                        [z, 640+128, 640-128],  # bottom-left
            ]
            roi_box.append(rect_seg)

        self._viewer.add_shapes(
            roi_box,
            shape_type='rectangle',
            edge_color='green',
            face_color='transparent',
            edge_width=1,
            name='ROI'
        )
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
            self.input_image_dropdown.addItems(input_layers)
            self.input_image_dropdown.setEnabled(True)


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

    def save_selected_prediction(self):
        selected_layer_name = self.layer_combo.currentText()
        if not selected_layer_name or selected_layer_name == "No result layer found":
            QtWidgets.QMessageBox.warning(self, "Warning", "No result layer selected.")
            return

        layer = self._viewer.layers[selected_layer_name]
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return

        save_path = os.path.join(output_dir, f"{selected_layer_name}.npy")
        np.save(save_path, layer.data)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {save_path}")




