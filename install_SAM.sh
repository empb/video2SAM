# DO NOT FORGET: environment variable CUDA_LAUNCH_BLOCKING must be set to 1.
# (Will be set by *.py scripts themselves).

if pip show segment-anything > /dev/null
then
    echo "##### Python package segment-anything already installed. #####"
else
    echo "##### Installing package segment-anything... #####"
    pip install git+https://github.com/facebookresearch/segment-anything.git
    echo "##### ... package segment-anything installed. #####"
fi