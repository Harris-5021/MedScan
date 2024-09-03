# MedScan: AI-Powered Pneumonia Detection

MedScan is an advanced web application that uses artificial intelligence to analyse chest X-ray images and detect the presence of pneumonia.
This system employs a trained neural network model to provide fast, accurate diagnoses, assisting healthcare professionals in their decision-making process.

## Features

- **AI-Powered Analysis**: Utilises a state-of-the-art neural network model to analyze chest X-rays.
- **Pneumonia Detection**: Accurately identifies the presence of pneumonia in uploaded X-ray images.
- **Probability Score**: Provides a probability percentage indicating the likelihood of pneumonia.
- **User-Friendly Interface**: Easy-to-use web interface for uploading and analyzing X-ray images.
- **Responsive Design**: Accessible on various devices, from desktop to mobile.

## How It Works

1. **Upload**: Users upload a chest X-ray image through the web interface.
2. **Analysis**: Our AI model processes the image using advanced computer vision techniques.
3. **Results**: The system displays the analysis results, including:
   - Detection of pneumonia (positive or negative)
   - Probability score indicating the likelihood of pneumonia

## Technology Stack

- Frontend: HTML, CSS, JavaScript, React
- Backend: ASP.NET Core
- AI Model: TensorFlow.js
- Database: SQL.

## Installation and Setup
- Need to have .net8 installed
- This project uses an mvc model
- database can be used locally e.g. sql server management studio

## Usage

1. Navigate to the MedScan homepage.
2. Register an account
3. Login to account
4. Click on the "Analyse X-ray" button.
5. Upload a chest X-ray image.
6. Wait for the analysis to complete.
7. View the results, including the pneumonia detection and probability score.

## Contributing

We welcome contributions to improve MedScan.

## License

No License currently, subject to change.

## Acknowledgments

- ai model was trained on a dataset from kaggle https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Disclaimer

MedScan is designed as a supportive tool for healthcare professionals and is only a personal project developed by a non-professional. This application
should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
