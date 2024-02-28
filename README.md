# Predicting the name of the Kraków street based on the photo


## Intro
The idea was to train a model that would be able to classify a photo and return the street name in Kraków. The city is full of tenement houses and every one is unique, so the model should learn how to distinguish them (as different parts of the city have been built at different ages - the city is about 1000 years old!)
The dataset has been collected with the Google Street View API (the free tier allows max 640x640 resolution).


## Usage
You need a Google [Streetview API access](https://developers.google.com/maps/documentation/streetview/overview) (free tier).
**Keep an eye on your API usage limits!**

Modify the config.ini file with your variables

Set env variable with your Google API key
```bash
export GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

The images can be downloaded with the scraper
```python3
python3 scraper/google_streetview_scraper.py
```

The model can be trained with train.py
```python3
python3 train/train.py
```

A simple Flask app has been built to serve as a GUI where you can load your own photo and get the street name in return.
```python3
python3 app/app.py
```

## Preview
![image](https://github.com/piotrb9/street-predictor/assets/157641773/0efb4830-11f6-4d84-b78c-40a820fad7fa)
![image](https://github.com/piotrb9/street-predictor/assets/157641773/6b4c04af-d946-4e4f-9d67-f132e0946a24)

### **LIVE DEMO: http://18.159.250.200/**
Hosted on AWS EC2.


## Tools
<img src="https://icon.icepanel.io/Technology/svg/Pandas.svg" width="100" height="100"><img src="https://icon.icepanel.io/Technology/svg/PyTorch.svg" width="100" height="100"><img src="https://icon.icepanel.io/Technology/svg/Python.svg" width="100" height="100"><img src="https://icon.icepanel.io/Technology/svg/PyCharm.svg" width="100" height="100"><img src="https://icon.icepanel.io/Technology/svg/AWS.svg" width="100" height="100">

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Resources
Using Google Street View API.

Made with the help of
- https://towardsdatascience.com/pytorch-image-classification-tutorial-for-beginners-94ea13f56f2

## License

[MIT](https://choosealicense.com/licenses/mit/)
