# BANA 7075: Machine Learning System Design for Business
# Final Project: Used Car Price Estimator 🚗
### Team 4: Joey Garascia, Dontrell Johnson, Tricia McHale, Soham Patel, Bella Rodino

#### The Used Car Price Estimator is a machine-learning tool that provides data-driven price estimates for used vehicles, reducing information asymmetry between buyers and sellers in the used car market.

#### Data Set
- [Used Car Sales Synthetic Dataset from Kaggle](https://www.kaggle.com/datasets/sandeep1080/used-car-sales)
- 10,000 AI-generated records spanning from 2015 to 2024, with 25 columns

##### Column Descriptions:
- ID: Unique identifier for each record.
- Distributor Name: Name of the car distributor.
- Location: Location of the distributor’s office.
- Car Name: The specific name of the car.
- Manufacturer Name: Name of the car’s manufacturer.
- Car Type: Type of car (e.g., Sedan, SUV, Hatchback, etc.).
- Color: Car’s color.
- Gearbox: Type of gearbox (e.g., Manual, Automatic).
- Number of Seats: Total number of seats in the car.
- Number of Doors: Number of doors in the car.
- Energy: Fuel type used by the car (e.g., Petrol, Diesel, Electric).
- Manufactured Year: Year the car was manufactured.
- Price-$: Listed price of the car in USD.
- Mileage-KM: Total kilometers the car has traveled.
- Engine Power-HP: Horsepower (HP) of the car’s engine.
- Purchased Date: Date the distributor purchased the car.
- Car Sale Status: Indicates whether the car was sold to a customer (Sold/Not Sold).
- Sold Date: Date the car was sold to a customer.
- Purchased Price-$: Purchase price paid by the distributor.
- Sold Price-$: Sale price paid by the customer.
- Margin-%: Percentage margin earned by the distributor.
- Sales Agent Name: Name of the sales agent who closed the deal.
- Sales Rating: Rating given to the sales agent by the distributor.
- Sales Commission-$: Commission paid to the sales agent by the distributor.
- Feedback: Customer feedback on the sales experience.

## Tech Stack
- **Model**: Random Forest / XGBoost (scikit-learn)
- **Experiment Tracking**: MLflow
- **Backend API**: FastAPI
- **Frontend**: Streamlit
- **Version Control**: Git + GitHub
- **Data Versioning**: DVC
