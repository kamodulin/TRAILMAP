from training import *
from training.label_processor import process_labels
from utilities.utilities import *

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")

    # Must include operation being done.
    # e.g. generate_data_set
    if len(sys.argv) < 2:
        raise Exception("You must include the type of preparation. Please refer to readme")

    # Generate data set by cropping out 64 length cubes from larger chunks for validation
    if sys.argv[1] == "generate_validation_set":

        # Validation directories
        data_original_path = base_path + "/data/validation/validation-original"
        data_set_path = base_path + "/data/validation/validation-set"

        # Default value is None
        nb_examples = None
        if len(sys.argv) > 2:
            nb_examples = int(sys.argv[2])

        generate_data_set(data_original_path, data_set_path, nb_examples=nb_examples)

    # Generate data set by cropping out 64 length cubes from larger chunks for training
    elif sys.argv[1] == "generate_training_set":

        # Training directories
        data_original_path = base_path + "/data/training/training-original"
        data_set_path = base_path + "/data/training/training-set"

        # Default value is None
        nb_examples = None
        if len(sys.argv) > 2:
            nb_examples = int(sys.argv[2])

        generate_data_set(data_original_path, data_set_path, nb_examples=nb_examples)

    # Add edge labels to your labeled data
    elif sys.argv[1] == "process_labels":

        if len(sys.argv) < 3:
            raise Exception("You must include the directory to the labels")

        labels_folder = sys.argv[2]
        output_name = "processed-labels"
        output_dir = os.path.dirname(labels_folder)
        output_folder = os.path.join(output_dir, output_name)

        if not os.path.isdir(labels_folder):
            raise Exception(labels_folder + " is not a directory. Inputs must be a folder of tiff files. Please refer to readme for more info")

        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

        for label_name in os.listdir(labels_folder):
            vol = read_tiff_stack(os.path.join(labels_folder, label_name))
            edge_vol = process_labels(vol)
            write_tiff_stack(edge_vol, os.path.join(output_folder, label_name))

    # Generates random 3D subvolumes of raw data for annotation
    elif sys.argv[1] == "generate_annotation_subvolumes":
        if len(sys.argv) < 4:
            raise Exception("You must include input and output directories")

        raw_dir = sys.argv[2]
        output_dir = sys.argv[3]
        
        # Default value for lateral edges is 200px
        cube_length = 200
        if sys.argv[4]:
            cube_length = int(sys.argv[4])

        if not os.path.isdir(raw_dir):
            raise Exception(raw_dir + " is not a directory. Inputs must be a folder of tiff files. Please refer to readme for more info")

        if not os.path.isdir(output_dir):
            raise Exception(output_dir + " is not a directory. Inputs must be a folder of tiff files. Please refer to readme for more info")

        coords = [('file', 'x', 'y')]

        for raw_vol in os.listdir(raw_dir):
            vol = read_tiff_stack(os.path.join(raw_dir, raw_vol))

            z, x, y = vol.shape
            shape = (cube_length, cube_length, z)

            x = np.random.randint(0, x-cube_length)
            y = np.random.randint(0, y-cube_length)

            sub_vol = crop_box(x, y, 0, vol, shape)
            write_tiff_stack(sub_vol, os.path.join(output_dir, raw_vol))
            
            coords.append((raw_vol, x, y))
        
        np.savetxt(os.path.join(raw_dir, "crop-coords.csv"), coords, fmt='%s', delimiter=",")

    else:
        raise Exception("You must choose your type of preparation (generate_validation_set, generate_training_set, process_labels)")
