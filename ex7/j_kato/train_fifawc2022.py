import json

config_path = "./content/config/exp_fifawc2022.json"

with open(config_path, 'r') as f:
    config_data = json.load(f)

print(config_data)

# Modify some values in the config_data dictionary
config_data['dataset']['train_filename'] = './data/fifawc/observation_data/split/train'
config_data['dataset']['valid_filename'] = './data/fifawc/observation_data/split/validation'
config_data['dataset']['test_filename'] = './data/fifawc/observation_data/split/test'

print(f"config: {config_data['dataset']['test_filename']}")

# changed learning rate and max epoch
print(f"max epochs: {config_data['max_epochs']}, learning rate: {config_data['model']['optimizer']['lr']}")

# Modify some values in the config_data dictionary
config_data['max_epochs'] = 5  # Example: Change max_epochs to 30
config_data['model']['optimizer']['lr'] = 0.01 # Example: Change learning rate to 0.001
config_data['datamodule']['type'] = 'rl_attacker' # Fix the datamodule type

# You can modify other values as needed
# config_data['datamodule']['batch_size'] = 128

# Print the modified values to confirm
print(f"max epochs: {config_data['max_epochs']}, learning rate: {config_data['model']['optimizer']['lr']}")
print(f"datamodule type: {config_data['datamodule']['type']}")

with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=4)

print(f"Modified config saved to {config_path}")