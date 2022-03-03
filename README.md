<!-- Add banner here -->

# Taxi Fare Prediction Model

<!-- Add buttons here -->
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)

This aim of this project is to train a model at scale using the [Kaggle New York City Taxi Fare Dataset](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data) and predict the price of a new taxi journey. I will host an API using Google Cloud Run on a lightweight website using Streamlit and Heroku. The repo that stores the code for the front-end is [here](https://github.com/kai-majerus/TaxiFareWebsite). 

The website can be found [here](https://kmajerus-taxifareapi.herokuapp.com/) and is a work in progress. I would like to add a map interface so that users can select pickup and dropoff locations on the map rather than entering longitude and latitude.

Tech stack
* Language - Python
* Tools - GCP, ML Flow, Streamlit, Heroku.
* Libraries - Pandas, NumPy, sklearn

# Demo-Preview
[(Back to top)](#table-of-contents)

To Add

# Table of contents

- [Project Title](#taxi-fare-prediction-model)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Startup the project](#startup-the-project)
- [Installation](#installation)
- [Development](#development)
- [Footer](#footer)

# Startup the project
[(Back to top)](#table-of-contents)

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for TaxiFareModel in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/TaxiFareModel`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "TaxiFareModel"
git remote add origin git@github.com:{group}/TaxiFareModel.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```

# Install

Go to `https://github.com/{group}/TaxiFareModel` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/TaxiFareModel.git
cd TaxiFareModel
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```

# Development
[(Back to top)](#table-of-contents)

To Add

# Footer
[(Back to top)](#table-of-contents)

![footer_video](https://user-images.githubusercontent.com/53292276/156608882-fd58c52c-6aec-4710-9544-54529ba4eba0.gif)
