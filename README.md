# MTech-Thesis
Repository for the work during my mtech thesis

## LEAD Dataset

This dataset contains hourly electricity meter readings and anomaly annotations for various commercial buildings over a period of up to one year. The data is structured as follows:

- `building_id`: Unique identifier for each building.
- `timestamp`: Hourly timestamp of the meter reading.
- `meter_reading`: Actual electricity meter reading value.
- `anomaly`: Binary indicator of whether the timestamp (reading) is considered anomalous (1) or not (0).

The dataset covers readings from 200 buildings, with each building having approximately 8,747 data points. Anomaly annotations are provided to mark specific timestamps within each building's time series where anomalous readings were detected.

Here's a small example of the dataset:

| building_id | timestamp       | meter_reading | anomaly |
|-------------|-----------------|---------------|---------|
| 1           | 01-01-2016 00:00| 100.5         | 0       |
| 1           | 01-01-2016 01:00| 98.2          | 0       |
| 1           | 01-01-2016 02:00| 95.7          | 0       |
| 2           | 01-01-2016 00:00| 200.1         | 0       |
| 2           | 01-01-2016 01:00| 203.4         | 1       |
| 2           | 01-01-2016 02:00| 197.8         | 0       |

## Config JSON File Details

Given below is the config file with default values.

```yaml
{
    "data": {
        "dataset_path": "dataset/15_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": true,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": false,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```

### Data Section
- `dataset_path`: Path to the dataset file (`"dataset/15_builds_dataset.csv"`) (Needs to be uploaded)
- `train_path`: Path where the training data or model inputs are stored (`"model_input/"`)
- `only_building`: Particular building identifier or index (`1304`)

### Training Section
- `batch_size`: Number of samples per batch during the training process (`128`)
- `num_epochs`: Number of training epochs (`200`)
- `latent_dim`: Dimensionality of the latent space in the model (`100`)
- `w_gan_training`: Indicates whether to use Wasserstein GAN (WGAN) training (`true`)
- `n_critic`: Number of critic iterations per generator iteration in WGAN training (`5`)
- `clip_value`: Clipping value for the critic's weights in WGAN training (`0.01`)
- `betaG` and `betaD`: Beta values for the generator and discriminator, respectively (`0.5`)
- `lrG` and `lrD`: Learning rates for the generator and discriminator, respectively (`0.0002`)

### Preprocessing Section
- `normalize`: Indicates whether to normalize the sements (transform all the readings in a segment to be in the [-1,1] range). (`true`)
- `plot_segments`: Specifies whether to plot the segments (`true`)
- `store_segments`: Indicates whether to store the segments (`true`)
- `window_size`: Size of the window for data preprocessing (`48`)

### Reconstruction (recon) Section
- `use_dtw`: Will work on this later(`false`)
- `iters`: Number of iterations  used by the gradient descent algorithm in noise space for rconstruction (`1000`)
- `use_eval_mode`: Indicates whether to use evaluation mode of the Generator is used during reconstruction (`true`)

## Anomaly Detection on the Entire Dataset

Anomaly detection process for the entire set of 200 buildings follow the same steps. Each building gets its own GAN model. The process is automated by the `run.py` script where the reconstruction pickle files are obtained for each building by running the `preprocessing.py`, `training.py` and `testing.py` scripts on loop.

1. Set up the appropriate configuration in `config.json`
2. Run `run.py` (It runs 3 scripts and creates reconstruction data pickle files)
3. Run `anom_detect_gan.py` 
4. Run `plotting.py` to create plots for the anomaly detection


## Baseline Methodologies


The directory "experimental" contains code for comparisons with other popular methods. We perform anomaly detection using different methodologies and also try to maintain similar evaluation and training hyper-parameters for fair comparisons.
