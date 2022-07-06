# Ionosphere storms

This is an app I created as part of my master's thesis. I created an independent platform to warn about the occurrence of ionospheric storms in Wroclaw and Lower Silesia in general. Calculations are performed on the basis of files received from WROC station, and all visualizations were made using Plotly and Leaflet libraries. 

Below is a diagram of how to do the calculations
![Schemat_obliczen](https://user-images.githubusercontent.com/69639195/177517542-0cf23d12-a58b-4e6e-8e69-a176316145f5.png)

Schemat stworzonej aplikacji
![aplikacja_struktura](https://user-images.githubusercontent.com/69639195/177517975-2096531e-704f-490e-af63-9b6bf6f2d910.png)

The application provides a number of maps and charts, including charts of pseudo-distance values for GPS and Galileo and the relationship between sTEC and vTEC values and elevation angle and azimuth. However, the main product is a map showing sTEC and vTEC values projected onto the Earth's surface to predict the arrival of ionospheric storms, which affect, among other things, the accuracy of the received GPS signal and telecommunications. 

![vTEC_map](https://user-images.githubusercontent.com/69639195/177519285-f9edc053-4e38-4a3a-ac58-689a25f8f5bf.jpg)
