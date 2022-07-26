# VSS(Video Semantic Segmentation)

## To setup the required python environment

### Windows Command Line

<details>
<summary>Fold/Unfold venv guide in Windows Command Line</summary>

1. Make virtual environment using venv
   `python -m venv vss`

2. Change the current directory to vss
   `cd vss`

3. Activate the virtual environment
   `Scripts\activate.bat`
   
4. The result will be something like this
   (vss) C:\project\vss>

</details>

### Windows Powershell

<details>
<summary>Fold/Unfold venv guide in Windows Powershell</summary>

1. Make virtual environment using venv
   `python -m venv vss`

2. Change the current directory to vss
   `cd vss`

3. Activate the virtual environment
   `./Scripts/Activate.ps1`
   
4. The result will be something like this
   (vss) C:\project\vss>

</details>

### Mac/Linux

<details>
<summary>Fold/Unfold venv guide in Mac/Linux</summary>

1. Make virtual environment using venv
   `python -m venv vss`

2. Change the current directory to vss
   `cd vss`

3. Activate the virtual environment
   `source ./bin/activate`
   
4. The result will be something like this
   (vss) ...\vss$

</details>

- To install the required packages
  - `pip install -r requirements.txt`
- To deactivate the virtual environment
  - `deactivate`
- To run the model
  - `python features_logits_extractor.py`

## To build annotated csvs from annotations

- `python annotation.py --save`

## To test temporal non-maximum suppression

- `python test_tnms.py`

## To run the S3D model with pretrained weights to classify a video

1. Convert a video file into several frames
- `python video2images.py`

2. Run the S3D model
- `python features_logits_extractor.py`