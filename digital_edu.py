import pandas as pd
#----------------------------------Крок 1: завантаження та очищення даних----------------
df = pd.read_csv('train.csv')
print(df.head())
print(df.info())

#----------1. Видаляємо непотрібні стовпчики----------
df.drop(['id', "bdate", "has_photo","has_mobile", "city", "followers_count", "last_seen", "occupation_name", "education_status"], axis=1, inplace=True)



#----------2. В стовпчику "sex" нема 0. Тільки так: 1-жінка, 2-чоловік. Перетворюємо 1 в 0, а 2 в 1----------
def fill_sex(sex):
    if sex == 1:
        return 0
    return 1
df["sex"] = df["sex"].apply(fill_sex)



#----------3. Розбираємось зі стовпчиком "education_form"----------
#дивимось які форми навчання користувачів найпоширеніші
print(df["education_form"].value_counts())

#заповнюємо порожні значення найпоширенішим значенням - "Full-time"
df['education_form'].fillna("Full-time", inplace=True)

#створюємо категоріальні змінні для кожної форми навчання людини
df[list(pd.get_dummies(df["education_form"]).columns)] = pd.get_dummies(df["education_form"])
df.drop("education_form", axis=1, inplace=True)



# ----------4. Розбираємось зі стовпчиком "relation"----------
# дивимось які сімейні сановища користувачів найпоширеніші
print(df["relation"].value_counts())

#заповнюємо порожні значення найпоширенішим значенням - 0
df['relation'].fillna(0, inplace=True)



#----------5. Розбираємось зі стовпчиком "langs"----------
#функція, яка розділяє рядок з двокрапками на список 


def split_langs(genres):
   return genres.split(';')

#переписуємо стовпичк 'langs' - представляємо його елементи у вигляді списку
df['langs'] = df['langs'].apply(split_langs)

#створюємо новий стовпчик 'number of langs' - тут показиватимуться к-ть мов, якими володіє людина
df['number of langs'] = df['langs'].apply(len)

#видаляємо стовпчик "langs" (він більше не потрібен)
df.drop("langs", axis=1, inplace=True)



#----------6. Розбираємось зі стовпчиком "life_main"----------
#якщо статус не "False", то конвертуємо його в число, інакше пишемо замість "False" число 0.
def life_main_int(status):
    if status != "False":
        return int(status)
    else:
        return 0
    
#перетворюємо дані в ствопчику "life_main" у числа
df['life_main'] = df['life_main'].apply(life_main_int)



#----------6. Розбираємось зі стовпчиком "people_main"----------
#якщо статус не "False", то конвертуємо його в число, інакше пишемо замість "False" число 0.
def people_main_int(status):
    if status != "False":
        return int(status)
    else:
        return 0
    
#перетворюємо дані в ствопчику "people_main" у числа
df['people_main'] = df['people_main'].apply(people_main_int)



# #----------7. Розбираємось зі стовпчиком "occupation_type"----------
#дивимось яке поточні заняття користувачів найпоширеніше
print(df['occupation_type'].value_counts())

#заповнюємо порожні значення найпоширенішим значенням - "university"
df['occupation_type'].fillna("university", inplace=True)

#створюємо категоріальні змінні для кожного заняття користувача
df[list(pd.get_dummies(df["occupation_type"]).columns)] = pd.get_dummies(df["occupation_type"])

#видаляємо стовпчик "occupation_type" (він більше не потрібен)
df.drop("occupation_type", axis=1, inplace=True)



#----------8. Розбираємось зі стовпчиком "career_start"----------
def career_start_int(status):
    if status != "False":
        return int(status)
    else:
        return 2024
#перетворюємо дані в ствопчику "career_start" у числа
df['career_start'] = df['career_start'].apply(career_start_int)

#створюємо новий стовпчик 'career_duration' - тут показиватиметься скільки років вже людина працює
def find_career_duration(year):
    return 2024 - year
df['career_duration'] = df['career_start'].apply(find_career_duration)

#видаляємо стовпчик "career_start" (він більше не потрібен)
df.drop("career_start", axis=1, inplace=True)


#----------9. Розбираємось зі стовпчиком "career_end"----------
# print(df["career_end"].value_counts())
def career_end_int(status):
    if status != "False":
        return int(status)
    else:
        return 2024
#перетворюємо дані в ствопчику "career_start" у числа
df['career_end'] = df['career_end'].apply(career_end_int)

#створюємо новий стовпчик 'finish_career_duration' - тут показиватиметься скільки років вже людина працює
def find_finish_career_duration(year):
    return 2024 - year
df['finish_career_duration'] = df['career_end'].apply(find_finish_career_duration)

#видаляємо стовпчик "career_end" (він більше не потрібен)
df.drop("career_end", axis=1, inplace=True)



# ----------------------------------------------Крок 2: Створення моделі-------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#беремо дані, на основі яких ми будемо прогнозувати результат (все крім result)
x = df.drop('result', axis = 1)
#беремо дані, які будуть прогнозуватися (result)
y = df['result']

#ділимо дані на 2 набори: тренувальний та тестувальний
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Стандартизуємо дані(як тренувальні, так і тестувальні)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#створюємо математичну модель k-найближчих сусідів та "згодовуємо" їй тренувальні дані
our_model = KNeighborsClassifier(n_neighbors = 499)
our_model.fit(x_train, y_train)

#просимо модель спрогнозувати результати на основі тестових даних
y_pred = our_model.predict(x_test)
print('Відсоток правильно передбачених результатів:', accuracy_score(y_test, y_pred) * 100)


