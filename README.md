# Sakura_season_in_Tokyo
Predicting the cherry blossom season in Tokyo

In this project an approach similar to that in project "Cherry blossom dates in Hirosaki park" is used to predict the begin, height and end of cherry blossom season in Tokyo. 
For the prediction the daily avarege temperatures from March 1. to March 10. are used. The trained GBRegressor from the Hirosaki project is retrained on historical Tokyo temperature data and used to predict the temperatures from March 10 until the end of May. The predicted temperatures are then used as input for a neural net to predict only the start date or start, height and end of the cherry blossom season. 

The csv.-files were created using the data from https://www.wunderground.com/history/monthly/jp/tokyo/RJTT for historical temperature and https://www.japan-guide.com/blog/sakura10/100323_tokyo.html for cherry blossom status. To get the temperatures for the prediction this year following site was used https://www.accuweather.com/en/jp/tokyo/226396/march-weather/226396?year=2022 .

The predicted dates are: 
start: '16.3', height: '5.4', end: '10.4'

Prediction of the start date only: 
start: 17.03. 
