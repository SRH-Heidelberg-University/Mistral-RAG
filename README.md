# Mistral Internal Search

SRH-CHATBOT is a sophisticated chatbot system tailored for university environments, designed to deliver efficient and precise responses by utilizing cutting-edge technologies such as Meta Llama2, Mistral and Pinecone. Engineered for simplicity and scalability, it is ideally suited for diverse university-related applications, from streamlining administrative tasks and improving operational efficiency to facilitating interactive learning environments for students and faculty. This chatbot aims to transform communication within the university, ensuring that information and academic support are more accessible and effective for the entire university community.

## Getting Started
To get a working instance of Mistral Internal Search for development and testing purposes, follow the below instructions.

### Prerequisites
- **Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).
- **Pinecone Account**: Sign up at [Pinecone.io](https://www.pinecone.io/).

### Setup Instructions

#### Step 1: Clone the Repository
Use the following command to clone the Mistral repository :
```bash
git clone https://github.com/Tejasree-Reddy/Mistral_Internal_Search.git
```

#### Step 2: Create a Conda Environment
Navigate to the project directory and use Python 3.8 to build a new Conda environment called `nenv`
```bash
conda create -n nenv python=3.8 -y
```
Activate the environment:
```bash
conda activate nenv
```

#### Step 3: Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Step 4: Setup Pinecone account
Create an account in pinecone and create a Pinecone index named `casestudy` with the following specifications:
- **Metric**: Cosine
- **Dimensions**: 768

This Pinecone index is essential for storing and retrieving vectorized representations of your documents. Once the index is set up, follow the bellow instruction
- Copy the `API_key` from API Keys menu in the dashboard 
- If you want to update the pinecone index name in any case, rename `index` variable in "pinecone_key.py" and save the file


**Using Local PDF Documents**:
  A few textbooks are available locally in `Data` folder


**Using My Existed Account**:
  You can directly use my pinecone account. The necessary details have been encapsulated within the code.

**In near future, the current pinecone-client version used here may deprecate. The `commented` code supports the recent version(3.0.0) launched by pinecone is also included in the `vectordb.py` file. Before using it, run "pip install pinecone-client==3.0.0" in the nenv environment**

### Launch the LLM
Run the `output.py` file to verify the answers from mistral

### Tech Stack:
- Python: Core programming language
- LangChain: Library for building language model applications
- Mistral : AI model for natural language understanding and generation
- Pinecone: Vector database for similarity search

### Acknowledgments
Special thanks to SRH Heidelberg for supporting this project.
