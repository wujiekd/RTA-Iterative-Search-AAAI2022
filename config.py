args_wideresnet = { 
    'epochs': 180,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'weight_decay': 5e-4   
    },
    'scheduler_name': 'MultiStepLR',
    'scheduler_hyperparameters': {
        'milestones': [120,140,160],
        'gamma': 0.1
    },
    'batch_size': 128,
}
args_preactresnet18 = { 
    'epochs': 180,  
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'weight_decay': 5e-4   
    },
    'scheduler_name': 'MultiStepLR',
    'scheduler_hyperparameters': {
        'milestones': [120,140,160],
        'gamma': 0.1
    },
    'batch_size': 128,
}

