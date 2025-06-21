def get_config(dataset):
    dataset = dataset.lower()
    
    if dataset == 'mnist' or dataset == 'cifar10':
        return {
            'num_epochs': 350,
            'lr': 0.25,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'lr_milestones': [116, 233],
            'batch_size': 128
        }

    elif dataset == 'imagenet':
        return {
            'num_epochs': 300,
            'lr': 0.1,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'lr_milestones': [150, 225],
            'batch_size': 256
        }

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
