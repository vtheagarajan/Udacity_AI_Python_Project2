import argparse
import os

# Create the parser
my_parser = argparse.ArgumentParser(description='Train a machine learning model to identify images')

# Add the arguments
my_parser.add_argument('--checkpoint_file_with_path',
                       type=str,
                       help='the path to saved model checkpoint file')

my_parser.add_argument('--path_to_image',type=str,help='full path to image to predict')

# Execute the parse_args() method
args = my_parser.parse_args()

checkpoint_file_with_path = args.checkpoint_file_with_path if args.checkpoint_file_with_path is not None else 'checkpoint4.pth'
path_to_image = args.path_to_image 

print(os.path.isfile(os.path.join(os.curdir,checkpoint_file_with_path)))
