clear all

[SVMstr, C] = train_classifier();
classify_image('cat.jpg', SVMstr, C)