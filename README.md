## Face analysis

# Description
My project includes:
### AI Service:
+ **Fatigue detection:** detect whether the listener is tired, sleepy
+ **Attention Analysis:** Analyze attention level of viewer by tracking eyes gaze.
+ **Emotion Analysis:** Analyze the reaction level (positive/negative) of the viewer based on emotions (Happy, Neutral, Sad)

# Link model

1. Download ML model from [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Extract the model in `/Models` folder using:

    ```sh
    bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2 
    ```

# Prepare environment
```sh
$ virtualenv -p python3 python3
$ source python3/bin/active
$ pip install -r requirements.txt
```
# Run
```sh
# for run fatigue detect tired and sleepy add tracking eyes gaze
$ python FatigueTracking/run_fatigue.py

# for run detect emotions
$ python Emotion/run_emotion.py

```

# DEMOs

[Fatigue detection](https://drive.google.com/file/d/19SgNCdk8IRv8rK-mo9xaG1Hr87PhxI0v/view)

[Attention analysis](https://drive.google.com/file/d/1I8E_1JCE5oD4lr1xc2YijEGMXSUAQHH-/view)

[Emotion analysis](https://drive.google.com/file/d/1UbYNIJh3ZceSDasynRbTI-ItGtn8PvUY/view)
