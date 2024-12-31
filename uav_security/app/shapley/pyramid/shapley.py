import numpy as np
import torch
from openpyxl.styles.builtins import output
from torch.utils.data import Dataset
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from torch import nn
from typing import List, Tuple, Optional
import seaborn as sns
from app.models.pyramid.model.pyramid_conv_transformer import PyramidTransformer
from app.models.pyramid.model.pyramid_conv_lstm import PyramidConvLSTM
from app.models.pyramid.model.pyramid_conv_transformer_kan import PyramidTransformerKAN
from app.dataloader.dataloader_split import UAVDataset

class WrappedModel(nn.Module):
    def __init__(self, original_model, cutoff):
        super(WrappedModel, self).__init__()
        self.original_model = original_model
        self.cutoff = cutoff

    def forward(self, x):
        # Split the input into cyber and physical without unsqueezing
        cyber = x[:, :self.cutoff]          # [batch, cutoff]
        physical = x[:, self.cutoff:]       # [batch, physical_dim]
        return self.original_model(cyber, physical)

class SHAPAnalyzer:
    output_path  = os.path.dirname(__file__) + '/plots'
    output_force_path = output_path + '/force'
    output_top10_features = output_path + '/top10_features'
    output_shap_summarybar = output_path + '/output_shap_summarybar'
    output_shap_summarybeeswarm = output_path + '/output_shap_summarybeeswarm'
    output_decision_path = output_path + '/decision'
    output_partial_dependence_path = output_path + '/partial_dependence'
    output_interaction_heatmap = output_path + '/interaction_heatmap'
    output_feature_importance = output_path + '/feature_importance'
    output_cyber_physical_importance = output_path + '/cyber_physical_importance'
    def __init__(self,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 model_variant: str,
                 cutoff: int,
                 feature_names: Optional[List[str]] = None,
                 device: str = 'cpu',
                 sample_size: int = 100):
        """
        Initialize the SHAP Analyzer.
        """
        self.device = device
        self.model = WrappedModel(model, cutoff).to(device).eval()
        self.dataset = dataset
        self.sample_size = sample_size
        self.cutoff = cutoff
        self.model_variant = model_variant.lower()
        self.feature_names = feature_names
        self.class_names = ['DoS', 'benign', 'Replay', 'evil_twin']

        if self.feature_names is None:
            # Generate default feature names
            self.feature_names = [f'feature_{i}' for i in range(cutoff + self.dataset.input_dim_physical)]
        else:
            if len(self.feature_names) != (cutoff + self.dataset.input_dim_physical):
                raise ValueError("Number of feature names does not match total number of features.")

        # make dir
        # check path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.isdir(self.output_force_path):
            os.makedirs(self.output_force_path)
        if not os.path.isdir(self.output_top10_features):
            os.makedirs(self.output_top10_features)
        if not os.path .isdir(self.output_shap_summarybar):
            os.makedirs(self.output_shap_summarybar)
        if not os.path.isdir(self.output_shap_summarybeeswarm):
            os.makedirs(self.output_shap_summarybeeswarm)
        if not os.path.isdir(self.output_decision_path):
            os.makedirs(self.output_decision_path)
        if not os.path.isdir(self.output_partial_dependence_path):
            os.makedirs(self.output_partial_dependence_path)
        if not os.path.isdir(self.output_interaction_heatmap):
            os.makedirs(self.output_interaction_heatmap)
        if not os.path.isdir(self.output_feature_importance):
            os.makedirs(self.output_feature_importance)
        if not os.path.isdir(self.output_cyber_physical_importance):
            os.makedirs(self.output_cyber_physical_importance)


    def model_predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Model prediction function for SHAP.
        """
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values  # Convert DataFrame to NumPy array

        # print(f"model_predict input shape: {inputs.shape}")
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)  # [batch, total_features]
        with torch.no_grad():
            outputs = self.model(inputs_tensor)  # [batch, num_classes]
            # print(f"model_predict output shape: {outputs.shape}")
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

    def get_background_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get background and sample data for SHAP.
        """
        # Use a random subset of the dataset for background
        indices = np.random.choice(len(self.dataset), self.sample_size, replace=False)
        background = []
        for idx in indices:
            cyber, physical, _ = self.dataset[idx]
            cyber = cyber.numpy()
            physical = physical.numpy()
            instance = np.concatenate((cyber, physical))
            background.append(instance)
        background = np.array(background)  # [sample_size, total_features]

        # Similarly, select samples to explain
        sample_indices = np.random.choice(len(self.dataset), self.sample_size, replace=False)
        samples = []
        for idx in sample_indices:
            cyber, physical, _ = self.dataset[idx]
            cyber = cyber.numpy()
            physical = physical.numpy()
            instance = np.concatenate((cyber, physical))
            samples.append(instance)
        samples = np.array(samples)  # [sample_size, total_features]

        return background, samples

    def run_enhanced_shap_analysis(self, num_samples: int = 10, nsamples: int = 73):
        """
        Run enhanced SHAP analysis with additional plots and visualizations.
        """
        background, samples = self.get_background_samples()
        explainer = shap.Explainer(self.model_predict, background,
                                   output_names=self.class_names, output_type='probability')
        samples_to_explain = samples[:num_samples]
        shap_values = explainer(samples_to_explain)

        self.generate_feature_interaction_heatmap(shap_values)
        # self.generate_summary_plots(shap_values, samples_to_explain)
        # self.generate_decision_plots(shap_values, samples_to_explain)
        # self.generate_feature_importance_plots(shap_values)
        # self.generate_partial_dependence_plots(explainer, samples_to_explain)
        print("Enhanced SHAP analysis complete. Additional plots saved in the 'enhanced' subdirectory.")

    def generate_summary_plots(self, shap_values, samples_to_explain):
        for class_idx, class_name in enumerate(self.class_names):
            shap.summary_plot(
                shap_values[:, :, class_idx],
                samples_to_explain,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.title(f"SHAP Summary Plot (Bar) for {class_name}")
            plt.savefig(f'{self.output_shap_summarybar}/shap_summary_bar_{class_name}.png', bbox_inches='tight')
            plt.close()

            shap.summary_plot(
                shap_values[:, :, class_idx],
                samples_to_explain,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f"SHAP Summary Plot (Beeswarm) for {class_name}")
            plt.savefig(f'{self.output_shap_summarybeeswarm}/shap_summary_beeswarm_{class_name}.png', bbox_inches='tight')
            plt.close()

    def generate_decision_plots(self, shap_values, samples_to_explain):
        for i in range(len(samples_to_explain)):
            for class_idx, class_name in enumerate(self.class_names):
                print(f"Generating SHAP decision plot for sample {i}, class {class_name}")

                # Dynamically adjust feature names if needed
                num_features_in_shap = shap_values.values.shape[1]  # Number of features in SHAP values

                if len(self.feature_names) != num_features_in_shap:
                    print(
                        f"Adjusting feature names. Expected {num_features_in_shap}, but got {len(self.feature_names)}")
                    self.feature_names = [f'feature_{i}' for i in range(num_features_in_shap)]

                # Extract SHAP values for one sample and one class
                shap_val = shap_values.values[i, :, class_idx]  # [num_features]
                base_val = shap_values.base_values[i, class_idx]  # scalar

                # Generate decision plots
                shap.decision_plot(
                    base_val,
                    shap_val,
                    feature_names=self.feature_names,
                    show=False
                )
                plt.title(f"SHAP Decision Plot for Sample {i}, Class {class_name}")
                plt.savefig(f'{self.output_decision_path}/shap_decision_sample_{i}_class_{class_name}.png', bbox_inches='tight')
                plt.close()

    def generate_feature_importance_plots(self, shap_values):
        for class_idx, class_name in enumerate(self.class_names):
            feature_importance = np.abs(shap_values.values[:, :, class_idx]).mean(0)
            sorted_idx = np.argsort(feature_importance)
            sorted_features = [self.feature_names[i] for i in sorted_idx]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_features)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"Feature Importance for {class_name}")
            plt.tight_layout()
            plt.savefig(f'{self.output_feature_importance}/feature_importance_{class_name}.png')
            plt.close()

    def generate_feature_interaction_heatmap(self, shap_values):
        for class_idx, class_name in enumerate(self.class_names):
            # Extract the input data matrix for interaction computation
            samples_to_explain = shap_values.data  # Data used for SHAP explainer

            # Compute approximate interaction values for the first feature (index 0)
            shap_interaction_values = shap.approximate_interactions(0, samples_to_explain,
                                                                    shap_values[:, :, class_idx].values)

            # Ensure the shape is appropriate for heatmap (should be 2D)
            if shap_interaction_values.ndim == 1:
                shap_interaction_values = shap_interaction_values.reshape(-1, 1)

            # Normalize the interaction values to improve visibility in the heatmap
            shap_interaction_values = (shap_interaction_values - np.min(shap_interaction_values)) / (
                    np.max(shap_interaction_values) - np.min(shap_interaction_values) + 1e-9)

            # Plot interaction heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(shap_interaction_values,
                        xticklabels=self.feature_names,
                        yticklabels=self.feature_names,
                        cmap="coolwarm",
                        center=0)
            plt.title(f"Feature Interaction Heatmap for {class_name}")
            plt.tight_layout()
            plt.savefig(f'{self.output_interaction_heatmap}/feature_interaction_heatmap_{class_name}.png')
            plt.close()

    def generate_partial_dependence_plots(self, explainer, samples_to_explain, target_class=0):
        """
        Generate partial dependence plots for the specified target class.
        """
        for feature_idx, feature_name in enumerate(self.feature_names):
            print(
                f"Generating Partial Dependence Plot for feature {feature_name}, class {self.class_names[target_class]}")

            # Extract predictions for the specific class
            def model_class_predict(inputs):
                return self.model_predict(inputs)[:, target_class]  # Only take the predictions for the target class

            # Generate partial dependence plots for the chosen feature and class
            shap.partial_dependence_plot(
                feature_name,
                model_class_predict,  # Use the class-specific prediction function
                samples_to_explain,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f"Partial Dependence Plot for {feature_name} (Class: {self.class_names[target_class]})")
            plt.savefig(
                f'{self.output_partial_dependence_path}/partial_dependence_{feature_name}_class_{self.class_names[target_class]}.png',
                bbox_inches='tight')
            plt.close()

    def analyze_cyber_physical_contributions(self, shap_values):
        cyber_features = self.feature_names[:self.cutoff]
        physical_features = self.feature_names[self.cutoff:]

        for class_idx, class_name in enumerate(self.class_names):
            cyber_importance = np.abs(shap_values.values[:, :self.cutoff, class_idx]).mean()
            physical_importance = np.abs(shap_values.values[:, self.cutoff:, class_idx]).mean()

            plt.figure(figsize=(10, 6))
            plt.bar(['Cyber', 'Physical'], [cyber_importance, physical_importance])
            plt.title(f"Cyber vs Physical Feature Importance for {class_name}")
            plt.ylabel("Mean |SHAP value|")
            plt.savefig(f'{self.output_cyber_physical_importance}/cyber_physical_importance_{class_name}.png')
            plt.close()

    def analyze_top_features(self, shap_values, top_n=10):
        for class_idx, class_name in enumerate(self.class_names):
            feature_importance = np.abs(shap_values.values[:, :, class_idx]).mean(0)
            top_features = np.argsort(feature_importance)[-top_n:]

            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), feature_importance[top_features])
            plt.yticks(range(top_n), [self.feature_names[i] for i in top_features])
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"Top {top_n} Features for {class_name}")
            plt.tight_layout()
            plt.savefig(f'{self.output_top10_features}/top_{top_n}_features_{class_name}.png')
            plt.close()

    def run_shap_analysis(self, num_samples: int = 10, nsamples: int = 73):
        """
        Run SHAP analysis and generate plots.
        """


        # Get background and sample data
        background, samples = self.get_background_samples()

        # Initialize the SHAP Explainer with output_type specified
        explainer = shap.Explainer(self.model_predict, background,
                                   output_names=['DoS', 'benign', 'Replay', 'evil_twin'], output_type='probability')

        # Select a subset of samples to explain
        samples_to_explain = samples[:num_samples]  # [num_samples, total_features]

        # # Debugging: Print shapes
        # print(f"Background shape: {background.shape}")  # [sample_size, total_features]
        # print(f"Samples to explain shape: {samples_to_explain.shape}")  # [num_samples, total_features]

        # Compute SHAP values
        shap_values = explainer(samples_to_explain)

        # Debugging: Print overall shap_values structure
        # print(f"shap_values: {shap_values}")

        # Debugging: Print shapes of shap_values for each class
        class_names = self.class_names
        for class_idx, class_name in enumerate(class_names):
            if isinstance(shap_values[class_idx].values, np.ndarray):
                shap_val = shap_values[class_idx].values  # [num_samples, num_features]
                # print(f"SHAP values for class '{class_name}': {shap_val.shape}")
            else:
                print(f"SHAP values for class '{class_name}' have unexpected structure.")

        print(f"Number of feature names: {len(self.feature_names)}")

        # Fix: Choose SHAP values for one class (DoS as an example) for summary plots
        shap_val = shap_values[0].values  # Use SHAP values for the first class 'DoS'
        # print(f'shape_val.shape: {shap_val.shape}')
        # print(f'samples to explain shape:{samples_to_explain.shape}')
        # Generate summary plots for the selected class
        shap_val = shap_values[:, :, 0]
        shap.summary_plot(
            shap_val,
            samples_to_explain,
            feature_names=self.feature_names,
            show=False
        )


        plt.title(f"SHAP Summary Plot for {class_names[0]}")  # Adjust title to reflect the chosen class
        plt.savefig(f'{self.output_path}/shap_summary_{class_names[0]}.png', bbox_inches='tight')
        plt.close()

        # Generate force plots for individual samples
        for i in range(num_samples):
            for class_idx, class_name in enumerate(class_names):
                print(f"Generating SHAP force plot for sample {i}, class {class_name}")

                # Extract SHAP values for one sample and one class
                # Here, we extract SHAP values for class `class_idx`
                shap_val = shap_values.values[i, :, class_idx]  # [num_features]
                base_val = shap_values.base_values[i, class_idx]  # scalar
                sample = samples_to_explain[i]  # [num_features]

                # Ensure the length of features matches the SHAP values
                if len(shap_val) != len(sample):
                    raise ValueError(
                        f"Length of SHAP values ({len(shap_val)}) does not match the number of features ({len(sample)})!")

                # Generate force plots
                shap.force_plot(
                    base_val,
                    shap_val,
                    sample,
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )

                plt.title(f"SHAP Force Plot for Sample {i}, Class {class_name}")
                plt.savefig(f'{self.output_force_path}/shap_force_sample_{i}_class_{class_name}.png', bbox_inches='tight')
                plt.close()

        print("SHAP analysis complete. Plots saved as PNG files.")

    def explain_force_instance(self, instance_idx: int, nsamples: int = 100):
        """
        Generate SHAP force plots for a specific instance across all classes.
        """
        # Get the instance data
        cyber, physical, _ = self.dataset[instance_idx]
        cyber = cyber.numpy()
        physical = physical.numpy()
        instance = np.concatenate((cyber, physical)).reshape(1, -1)  # [1, total_features]
        instance_df = pd.DataFrame(instance, columns=self.feature_names)

        # Initialize explainer with background samples
        background, _ = self.get_background_samples()
        explainer = shap.Explainer(self.model_predict, background)

        # Compute SHAP values for the instance
        shap_values = explainer(instance_df)

        class_names = ['DoS', 'benign', 'Replay', 'evil_twin']
        for class_idx, class_name in enumerate(class_names):
            print(f"Generating SHAP force plot for instance {instance_idx}, class {class_name}")
            shap_val = shap_values[class_idx].values[0, :]  # [num_features]
            base_val = shap_values[class_idx].base_values[0]  # scalar
            sample = instance_df.iloc[0]
            shap.force_plot(
                base_val,
                shap_val,
                sample,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"SHAP Force Plot for Instance {instance_idx}, Class {class_name}")
            plt.savefig(f'shap_force_instance_{instance_idx}_class_{class_name}.png', bbox_inches='tight')
            plt.close()

        print(f"SHAP force plots for instance {instance_idx} generated and saved.")

    def run_comprehensive_analysis(self, num_samples: int = 10, nsamples: int = 73, top_n: int = 10):
        background, samples = self.get_background_samples()
        explainer = shap.Explainer(self.model_predict, background,
                                   output_names=self.class_names, output_type='probability')
        samples_to_explain = samples[:num_samples]
        shap_values = explainer(samples_to_explain)

        self.run_enhanced_shap_analysis(num_samples, nsamples)
        self.analyze_cyber_physical_contributions(shap_values)
        self.analyze_top_features(shap_values, top_n)

        print("Comprehensive SHAP analysis complete. All plots saved in the 'enhanced' subdirectory.")



if __name__ == '__main__':
    # Parameters
    data_path = '/data/fuse.csv'
    model_save_path = '/home/shengguang/PycharmProjects/uav_security/app/outputs/fuse_pyramid_conv_transformer_kan_WARMUP/models/best_pyramid_model.pth'
    cutoff = 56  # Ensure this matches your dataset configuration
    model_variant = 'transformer_kan'  # Options: 'transformer', 'transformer_kan', 'lstm'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the dataset
    dataset = UAVDataset(file_path=data_path, cutoff=cutoff)

    # Initialize the model based on the variant
    if model_variant == 'transformer':
        model = PyramidTransformer(input_dim_cyber=dataset.input_dim_cyber,
                                   input_dim_physical=dataset.input_dim_physical,
                                   num_classes=dataset.num_classes,
                                   num_layers=5,
                                   attention_heads=4,
                                   dropout=0.5)
    elif model_variant == 'transformer_kan':
        model = PyramidTransformerKAN(input_dim_cyber=dataset.input_dim_cyber,
                                      input_dim_physical=dataset.input_dim_physical,
                                      num_classes=dataset.num_classes,
                                      num_layers=5,
                                      attention_heads=4,
                                      dropout=0.5)
    elif model_variant == 'lstm':
        model = PyramidConvLSTM(input_dim_cyber=dataset.input_dim_cyber,
                             input_dim_physical=dataset.input_dim_physical,
                             num_classes=dataset.num_classes,
                             num_layers=5,
                             lstm_layers=4,
                             attention_heads=4)
    else:
        raise ValueError(f"Invalid model variant: {model_variant}")

    # Load the trained model weights
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded model weights from {model_save_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_save_path}")

    # Define feature names
    feature_names = [
        'frame.number', 'frame.len', 'frame.protocols', 'wlan.duration', 'wlan.ra',
        'wlan.ta', 'wlan.da', 'wlan.sa', 'wlan.bssid', 'wlan.frag', 'wlan.seq',
        'llc.type', 'ip.hdr_len', 'ip.len', 'ip.id', 'ip.flags', 'ip.ttl',
        'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
        'tcp.seq_raw', 'tcp.ack_raw', 'tcp.hdr_len', 'tcp.flags',
        'tcp.window_size', 'tcp.options', 'udp.srcport', 'udp.dstport',
        'udp.length', 'data.data', 'data.len', 'wlan.fc.type',
        'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status',
        'wlan.qos', 'wlan.qos.priority', 'wlan.qos.ack', 'wlan.ccmp.extiv',
        'wlan.wep.key', 'radiotap.hdr_length', 'radiotap.antenna_signal',
        'radiotap.signal_quality', 'radiotap.channel.flags.ofdm',
        'radiotap.channel.flags.cck', 'wlan_radio.datarate',
        'wlan_radio.channel', 'wlan_radio.frequency',
        'wlan_radio.signal_strength (dbm)', 'wlan_radio.Noise level (dbm)',
        'wlan_radio.SNR (db)', 'wlan_radio.preamble',
        'time_since_last_packet', 'mid', 'x', 'y', 'z', 'pitch', 'roll',
        'yaw', 'vgx', 'vgy', 'vgz', 'tof', 'h', 'battery', 'barometer',
        'agx', 'agy', 'agz'
    ]

    # Verify the number of feature names
    expected_num_features = dataset.input_dim_cyber + dataset.input_dim_physical
    if len(feature_names) != expected_num_features:
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match total number of features ({expected_num_features})")

    # Initialize SHAP Analyzer
    shap_analyzer = SHAPAnalyzer(
        model=model,
        dataset=dataset,
        model_variant=model_variant,
        cutoff=cutoff,
        feature_names=feature_names,
        device=device,
        sample_size=100  # Adjust based on your computational resources
    )

    # Run SHAP analysis
    # shap_analyzer.run_shap_analysis(num_samples=10, nsamples=73)
    shap_analyzer.run_comprehensive_analysis()

    # explain single instance for force plots
    # instance_idx = 0  # Replace with desired index
    # shap_analyzer.explain_instance(instance_idx=instance_idx, nsamples=73)