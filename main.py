import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
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

# Afficher les coefficients avec les noms des variables
feature_names = df.drop(columns=["charges"]).columns  # Récupérer les noms des variables
print("\nCoefficients du modèle :")
for feature, coef in zip(feature_names, theta):
    print(f"{feature}: {coef}")

# Graphique des coefficients
plt.figure(figsize=(10, 6))
plt.bar(feature_names, theta, color='skyblue')
plt.xlabel('Variables indépendantes')
plt.ylabel('Coefficients')
plt.title('Impact des variables sur les frais médicaux (Coefficients)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Validation croisée
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Scores validation croisée :", scores)
print("Moyenne des scores :", scores.mean())

y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Valeurs prédites')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Parfaite prédiction')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites (Scikit-Learn)")
plt.legend()
plt.grid()
plt.show()

#errors
mse = mean_squared_error(y_test, y_pred)
mae= mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print("\n MSE:",mse,"MAE:",mae,"R2:",r2)



# Déterminer le nombre de variables pour organiser les sous-graphiques
n_features = len(feature_names)
rows = (n_features // 3) + (n_features % 3 > 0)  # Nombre de lignes (max 3 colonnes)
cols = min(n_features, 3)  # Nombre de colonnes (max 3)


# Création des sous-graphiques
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()  # Aplatir pour un accès plus simple

# Boucle sur chaque variable indépendante
for i, feature in enumerate(feature_names):
    ax = axes[i]
    ax.scatter(X_test[:, i], y_test, color='blue', label='Réel', alpha=0.6)
    ax.scatter(X_test[:, i], y_pred, color='red', label='Prédit', alpha=0.6)
    ax.set_xlabel(feature)
    ax.set_ylabel('Charges')
    ax.set_title(f'{feature} vs Charges')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.7)

# Suppression des sous-graphiques vides si le nombre de variables < 9
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
