import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("frais_medicaux.txt", sep=",", encoding="utf-8")

# Prétraitement des données
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

# transformer catégories de région en colonnes binaires
df = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df.drop(columns=["charges"])
y = df["charges"] # On veut prédire les charges

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Régression linéaire avec Scikit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print("Score sur test-set =", model.score(X_test, y_test) * 100, '%')
print("Score sur train-set =", model.score(X_train, y_train) * 100, '%')

theta0 = model.intercept_
theta = model.coef_

print("Intercept (theta0) :", theta0)
print("Coefficients (theta) :", theta)

# Validation croisée
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Scores validation croisée :", scores)
print("Moyenne des scores :", scores.mean())

# Visualisation des résultats
plt.scatter(y_test, model.predict(X_test), color='blue')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites (Scikit-Learn)")
plt.grid()
plt.show()
