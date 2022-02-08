# Mengidentifikasi Faktor Dalam Menentukan Promosi Seorang Karyawan
## Latar Belakang 
Sebuah Perusahaan e-commerce yang sedang bertumbuh
memiliki 9 department di seluruh organisasi perusahaan, memiliki
78.298 karyawan yang di bagikan menjadi 3 level, yaitu Senior,
Middle dan Junior. Salah satu masalah yang dihadapi adalah
perusahaan tidak memiliki persyaratan yang spesifik dan pola
untuk mengidentifikasi karyawan yang sesuai untuk di promosikan
di waktu yang tepat. Kondisi saat ini, promosi dilakukan hanya
berdasarkan permintaan dari senior manager.
Untuk saat ini, perusahaan sedang membutuhkan bantuan
untuk mengidentifikasi faktor dalam menentukan promosi
seorang karyawan juga mempertimbangkan dari sisi Promotion
Cycle atau Siklus Promosi perusahaan tersebut. Beberapa data
telah dikumpulkan oleh perusahaan baik itu data performansi
lampau dan saat ini berdasarkan demografis.

Analisis ini memiliki beberapa bagian: 

1. Menyiapkan Library
      
      ```sh
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns
      from sklearn.preprocessing import LabelEncoder
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LogisticRegression
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.ensemble import GradientBoostingClassifier
      from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
      ```
2. Menyiapkan Dataset

      ```sh
      dataset=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQmDbYzIsdhTSGDJW-TNx3IBxob1JHqeTaeZkFSJjIq44oBUosLonZeDxkmAPIQirkGmN8f7S9R2q3x/pub?   gid=2043797390&single=true&output=csv')
      ```
3. Load Dataset
      
      ```sh
      dataset.head()
      ```
      
4. Penjelasan dari tiap variabel 
      - employee_id: Unique ID untuk Karyawan 
      - department: Department tempat Karyawan 
      - region: Region of employment (Tidak Teratur) 
      - education: Tingkat Pendidikan 
      - gender: Jenis Kelamin dari Karyawan 
      - job_level: Job Level di posisi terkini 
      - recruitmentchannel: Sumber recruitment karyawan 
      - no of trainings: Jumlah pelatihan yang pernah 
      diikuti 
      - age: Usia Karyawan 
      - previous year rating: Rating karyawan tahun 
      sebelumnya 
      - length of service: Lama kerja karyawan 
      - awards won?: Pernah mendapatkan penghargaan 
      ( ya : 1, Tidak : 0 ) 
      - avg_training_score: Nilai rata-rata pelatihan saat 
      ini 
      - satisfaction_score: Nilai kepuasan karyawan 
      terhadap perusahaan 
      - engagement_score: Nilai rata-rata karyawan 
      merasa keterikatan dengan Perusahaan 
      - is_promoted: Recommended for promotion

5. Prepocessing Data

      ```sh
      cleaned_dataset=dataset.copy()
      cleaned_dataset['education']=cleaned_dataset['education'].fillna('No Data')
      cleaned_dataset['previous_year_rating']=cleaned_dataset['previous_year_rating'].fillna(cleaned_dataset['previous_year_rating'].median())
      cleaned_dataset=cleaned_dataset.drop(['employee_id'], axis=1)
      cleaned_dataset=cleaned_dataset.rename(columns={'is_promoted':'label'})
      cleaned_dataset.isnull().sum()
      ```
      
6. Exploratory Data Analysis

      ```sh
      fig = plt.figure()
      ax = fig.add_axes([0,0,1,1])
      ax.axis('equal')
      labels = ['No', 'Yes']
      label = cleaned_dataset.label.value_counts()
      ax.pie(label, labels=labels, autopct='%.0f%%')
      plt.title('Percentage of employees who have been promoted')
      plt.show()
      ```
      
7. Machine Learning Model

      ```sh
      # Data preparation by encoding data and sharing train and test data for model building
      for column in cleaned_dataset.columns:
          if cleaned_dataset[column].dtype == np.number: continue
          cleaned_dataset[column] = LabelEncoder().fit_transform(cleaned_dataset[column])

      # Predictor dan target
      X = cleaned_dataset.drop('label', axis=1)
      y = cleaned_dataset['label']

      # Splitting train and test
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

      #SMOTE Oversampling Data Training
      from imblearn.over_sampling import SMOTE
      smote=SMOTE()
      x_train, y_train=smote.fit_resample(x_train, y_train)

      # Print according to the expected result
      print('The number of rows and columns of x_train is:', x_train.shape, ', while the Number of rows and columns of y_train is:', y_train.shape)
      print('The percentage of Promoted in the Training data is:')
      print(y_train.value_counts(normalize=True))
      print('The number of rows and columns of x_test is:', x_test.shape, ', while the Number of rows and columns of y_test is:', y_test.shape)
      print('The percentage of Promoted in the Testing data is:')
      print(y_test.value_counts(normalize=True))
      ```
