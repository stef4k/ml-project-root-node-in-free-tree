# ml-project-root-node-in-free-tree

## Installation and Execution
Follow these steps to set up the environment and jupyter kernel:

1. Create a virtual environment:
   ```sh
   python -m venv myenv
   ```

2. Activate virtual environment:
   ```sh
   myenv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Add the virtual environment as a new Jupyter kernel
   ```sh
   python -m ipykernel install --user --name=myenv --display-name "Python ML-project(myenv)"
   ```
