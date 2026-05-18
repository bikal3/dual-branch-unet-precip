# precipitation-downsampling
Project for ADLEO course.  Using a CNN to downscale IMERG precipitation data.

Project Summary
The goal of this project is to use a convolution neural network model to downscale IMERG near-real time precipitation data from 10 km resolution to 250 m resolution.  The area of focus is the Big Island of Hawaii.  The temporal frequency is daily over a 5 year period from 2015 to 2019. We will look at 2 different CNN architectures to compare accuracy.  The architectures are i) Dual-Branch U-Net, ii) Nested U-Net, and iii) DA-Net.

CNN inputs
- IMERG daily precipitation: | NASA IMERG Early Run V07B | Daily precipitation (mm) | 0.1° (~10 km) | Global | PPS HTTPS (auth required) |
- elevation: | SRTM/DEM | Elevation (m) | 30 m | Hawaii | Pre-downloaded |
- slope: | SRTM/DEM | Elevation (m) | 30 m | Hawaii | Pre-downloaded |
- aspect: | SRTM/DEM | Elevation (m) | 30 m | Hawaii | Pre-downloaded |
- cloud cover: | MODIS MOD09GA v061 (Terra) | Cloud state (state_1km_1 band) | 1 km | Global daily | NASA Earthdata LP DAAC |

CNN Target
-daily rainfall station: | HCDP Station CSVs | Daily rain gauge observations (mm) | Point (~165 stations) | Hawaii statewide | HCDP public API |

Loss Function: Mean Squared Error

Tasks:
-  Data aquisition code (Bikal)
-  Reproject precipitation and cloud cover to match extent(Big Island) and resolution of DEM (Elisabeth)
-  rasterize station data with non-station pixel masked (Gabby)
-  Dataloader, add rotation and flip augmentation, normalization (Elisabeth)
-  CNN1, Dual-Branch U-Net(Bikal)
-  CNN2, Nested U-Net(Gabby)
-  CNN3, DA-Net(Elisabeth)
-  Analysis of CNN with validation data, the year 2020. RMSE, MAE, Pearson(All)
-  Compare model output to HCDP interpolated rainfall map, difference maps, distribution, RMSE(All)
  
