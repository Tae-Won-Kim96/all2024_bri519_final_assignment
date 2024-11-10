# entrypoint.sh
#!/bin/bash

conda run -n myenv
python main.py
cp -r br517:./figure Docker_to_PNG