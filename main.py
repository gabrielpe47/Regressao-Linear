import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

dirpath = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(dirpath, 'Tabela.csv')

dataset = pd.read_csv(data)
spm = pd.DataFrame({'Sales': dataset['  Sales '], 'Month': dataset['Month Number'], 'Year': dataset['Year']})
spm = spm.sort_values(by=['Year', 'Month'], ascending=True)
spm['Sales'] = (spm['Sales']
                     .str.replace('$', '', regex=False)
                     .str.replace(',', '', regex=False)
                     .str.strip()
                     .astype(float))
print(spm)

x = spm['Month'].to_numpy().reshape(-1,1)
y = spm['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regression = LinearRegression()
regression = regression.fit(x_train, y_train)
prediction = regression.predict(x_test)

plt.bar(x_test.flatten() - 0.2, y_test, width=0.4, label='Real', color='blue')
plt.plot(x_test, prediction, color='red')
plt.title('Vendas por Mês')
plt.xlabel('Meses')
plt.ylabel('Vendas')
plt.show()
