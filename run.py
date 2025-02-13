"""
Main Entry Point
Executes complete A2D2 system workflow including data collection, training, and mission execution.
"""

import argparse
from pathlib import Path
import yaml

from sys_controller.process import ProcessController
from train_detector_yolov8 import train_yolo
from alert.alert_manager import AlertManager, AlertPriority, AlertType

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_data_collection(config: dict):
    """Run data collection phase."""
    controller = ProcessController(config)
    controller.collect_training_data(
        output_dir=config['collection']['output_dir']
    )

def run_training(config: dict):
    """Run model training phase."""
    train_yolo(
        data_yaml=config['training']['data_yaml'],
        model_name=config['training']['model_name'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        img_size=config['training']['img_size']
    )

def run_parameter_optimization(config: dict):
    """Run parameter optimization phase."""
    controller = ProcessController(config)
    controller.optimize_parameters()

def run_mission(config: dict):
    """Run complete mission with optimized parameters."""
    controller = ProcessController(config)
    controller.run_mission()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='A2D2 System Runner')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['collect', 'train', 'optimize', 'mission'],
        default='mission',
        help='Operation mode'
    )
    args = parser.parse_args()

    try:
        # Load configuration
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
        config = load_config(args.config)
        
        # Execute requested mode
        if args.mode == 'collect':
            run_data_collection(config)
        elif args.mode == 'train':
            if not Path(config['training']['data_yaml']).exists():
                raise FileNotFoundError(f"Data YAML not found: {config['training']['data_yaml']}")
            run_training(config)
        elif args.mode == 'optimize':
            run_parameter_optimization(config)
        else:
            run_mission(config)
            
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

