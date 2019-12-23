#!/bin/bash
docker run -it -p 8080:8080 -p 7000-7010:7000-7010 \
           -v ${PWD}:/workspace/ \
           mit6881/drake-course:pset4_2 /bin/bash -c "jupyter notebook ./pset4_pose_estimation.ipynb --ip 0.0.0.0 --port 8080 --allow-root --no-browser --NotebookApp.allow_origin='*' --NotebookApp.disabel_check_xsrf=True "
