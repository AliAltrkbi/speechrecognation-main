 Speech Recognition Using TensorFlow

This project implements a simple speech recognition model that classifies audio commands such as 'go', 'stop', 'left', 'right', and others using a dataset provided by TensorFlow. The model is built using TensorFlow and Keras.

 Prerequisites

Before running the code, you need to install the following dependencies:

1.Python 3.7+
2. Required Libraries:
   - `tensorflow`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `IPython`

You can install the dependencies by running the following command:


pip install tensorflow numpy matplotlib seaborn ipython


Steps to Run the Project

 1. Set Up the Project Environment

Make sure your environment is properly set up with the required libraries. You can use a virtual environment to isolate dependencies:

bash
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate
pip install -r requirements.txt  # Use this if you have a requirements file


 2. Download the Dataset

The dataset is automatically downloaded if it does not exist locally. The dataset used is the **Mini Speech Commands** dataset provided by TensorFlow.

The dataset consists of audio samples categorized into multiple commands such as 'go', 'stop', 'yes', 'no', and others.

 3. Run the Jupyter Notebook

To run the speech recognition code, simply launch the Jupyter Notebook:


jupyter notebook


Then open the `speechrecognation.ipynb` notebook.

 4. Execution Flow

1. Import the required libraries:
   The code imports libraries like `tensorflow`, `numpy`, `matplotlib`, and `seaborn` for building and visualizing the model.

2. Set a random seed:
   A random seed is set for reproducibility using TensorFlow and NumPy:

  
   seed = 42
   tf.random.set_seed(seed)
   np.random.seed(seed)


3. Load the Dataset:
   If the dataset is not already available, it will be automatically downloaded and extracted:


   data_dir = pathlib.Path('data/mini_speech_commands')
   if not data_dir.exists():
       tf.keras.utils.get_file(
           'mini_speech_commands.zip',
           origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
           extract=True,
           cache_dir='.', cache_subdir='data')


4. Display Available Commands:
   The code lists the available audio commands in the dataset:


   commands = np.array(tf.io.gfile.listdir(str(data_dir)))
   commands = commands[commands != 'README.md']
   print('Commands:', commands)


5. Dataset Overview:
   Prints the number of total examples and examples per label:


   filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
   filenames = tf.random.shuffle(filenames)
   num_samples = len(filenames)
   print('Number of total examples:', num_samples)
   print('Number of examples per label:', len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
   print('Example file tensor:', filenames[0])


5. Model Training

You can further customize the notebook to build and train a TensorFlow model for recognizing voice commands. The notebook already prepares the data for training, and you can use any model architecture to train it on this data.

 6. Visualizing Results

The notebook contains commands to visualize the results of the model training, such as accuracy and loss, using `matplotlib` and `seaborn`.

