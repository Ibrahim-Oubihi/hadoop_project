from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, year, hour, count, desc, dayofweek, weekofyear, expr
import matplotlib.pyplot as plt
import seaborn as sns

# Créer une session Spark
spark = SparkSession.builder.appName("AnalyseCriminaliteEtendue").getOrCreate()

# Configurer Spark pour utiliser le parseur de date legacy
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Lire le dataset depuis HDFS
dataset_file = (spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("hdfs://0.0.0.0:9000/user/hadoop/hadoop/crime.csv"))

# Convertir et extraire les informations de date
dataset_file = dataset_file.withColumn("date_rapport", to_date(col("date_of_report"), "MM/dd/yyyy HH:mm:ss a"))
dataset_file = dataset_file.withColumn("date_crime", to_date(expr("split(crime_date_time, ' - ')[0]"), "MM/dd/yyyy HH:mm"))
dataset_file = dataset_file.withColumn("mois_rapport", month("date_rapport"))
dataset_file = dataset_file.withColumn("annee_rapport", year("date_rapport"))
dataset_file = dataset_file.withColumn("jour_semaine", dayofweek("date_rapport"))
dataset_file = dataset_file.withColumn("heure_rapport", hour(col("date_of_report")))
dataset_file = dataset_file.withColumn("semaine_annee", weekofyear("date_rapport"))

# Analyses
crimes_par_type = dataset_file.groupBy("crime").agg(count("*").alias("count")).orderBy(desc("count")).limit(15)
crimes_par_quartier = dataset_file.groupBy("neighborhood").agg(count("*").alias("count")).orderBy(desc("count")).limit(15)
crimes_par_mois = dataset_file.groupBy("mois_rapport").agg(count("*").alias("count")).orderBy("mois_rapport")
crimes_par_jour = dataset_file.groupBy("jour_semaine").agg(count("*").alias("count")).orderBy("jour_semaine")
crimes_par_heure = dataset_file.filter(col("heure_rapport").isNotNull()).groupBy("heure_rapport").agg(count("*").alias("count")).orderBy("heure_rapport")
crimes_par_semaine = dataset_file.groupBy("semaine_annee").agg(count("*").alias("count")).orderBy("semaine_annee")
zones_rapport = dataset_file.groupBy("reporting_area").agg(count("*").alias("count")).orderBy(desc("count")).limit(15)
evolution_annuelle = dataset_file.groupBy("annee_rapport").agg(count("*").alias("count")).orderBy("annee_rapport")
type_crime_par_heure = dataset_file.filter(col("heure_rapport").isNotNull()).groupBy("heure_rapport", "crime").agg(count("*").alias("count")).orderBy(desc("count"))

# Conversion en listes pour le tracé
types_crimes = [row['crime'] for row in crimes_par_type.collect()]
comptes_types_crimes = [row['count'] for row in crimes_par_type.collect()]

quartiers = [row['neighborhood'] for row in crimes_par_quartier.collect()]
comptes_quartiers = [row['count'] for row in crimes_par_quartier.collect()]

mois = [row['mois_rapport'] for row in crimes_par_mois.collect()]
comptes_mois = [row['count'] for row in crimes_par_mois.collect()]

jours = [row['jour_semaine'] for row in crimes_par_jour.collect()]
comptes_jours = [row['count'] for row in crimes_par_jour.collect()]

heures = [row['heure_rapport'] for row in crimes_par_heure.collect()]
comptes_heures = [row['count'] for row in crimes_par_heure.collect()]

semaines = [row['semaine_annee'] for row in crimes_par_semaine.collect()]
comptes_semaines = [row['count'] for row in crimes_par_semaine.collect()]

zones = [row['reporting_area'] for row in zones_rapport.collect()]
comptes_zones = [row['count'] for row in zones_rapport.collect()]

annees = [row['annee_rapport'] for row in evolution_annuelle.collect()]
comptes_annees = [row['count'] for row in evolution_annuelle.collect()]

# Créer les graphiques
plt.figure(figsize=(25, 40))

# 1. Types de crimes
plt.subplot(5, 2, 1)
sns.barplot(x=types_crimes, y=comptes_types_crimes)
plt.title('Top 15 des types de crimes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Nombre')

# 2. Crimes par quartier
plt.subplot(5, 2, 2)
sns.barplot(x=quartiers, y=comptes_quartiers)
plt.title('Top 15 des quartiers par nombre de crimes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Nombre')

# 3. Crimes par mois
plt.subplot(5, 2, 3)
sns.lineplot(x=mois, y=comptes_mois, marker='o')
plt.title('Crimes par mois')
plt.xlabel('Mois')
plt.ylabel('Nombre')
plt.xticks(range(1, 13))

# 4. Crimes par jour de la semaine
plt.subplot(5, 2, 4)
sns.barplot(x=jours, y=comptes_jours)
plt.title('Crimes par jour de la semaine')
plt.xlabel('Jour de la semaine')
plt.ylabel('Nombre')
plt.xticks(range(7), ['Dim', 'Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam'])

# 5. Crimes par heure
plt.subplot(5, 2, 5)
sns.lineplot(x=heures, y=comptes_heures, marker='o')
plt.title('Crimes par heure')
plt.xlabel('Heure')
plt.ylabel('Nombre')
plt.xticks(range(0, 24, 2))

# 6. Crimes par semaine
plt.subplot(5, 2, 6)
sns.lineplot(x=semaines, y=comptes_semaines, marker='o')
plt.title('Crimes par semaine de l\'année')
plt.xlabel('Semaine')
plt.ylabel('Nombre')

# 7. Zones de rapport les plus fréquentes
plt.subplot(5, 2, 7)
sns.barplot(x=zones, y=comptes_zones)
plt.title('Top 15 des zones de rapport')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Nombre')

# 8. Évolution annuelle des crimes
plt.subplot(5, 2, 8)
sns.lineplot(x=annees, y=comptes_annees, marker='o')
plt.title('Évolution annuelle des crimes')
plt.xlabel('Année')
plt.ylabel('Nombre')

# 9. Types de crimes par heure (heatmap)
plt.subplot(5, 2, 9)
crime_hour_data = type_crime_par_heure.collect()
crime_hour_matrix = [[0 for _ in range(24)] for _ in range(len(types_crimes))]
for row in crime_hour_data:
    if row['crime'] in types_crimes and row['heure_rapport'] is not None:
        i = types_crimes.index(row['crime'])
        j = row['heure_rapport']
        crime_hour_matrix[i][j] = row['count']
sns.heatmap(crime_hour_matrix, cmap='YlOrRd', xticklabels=range(24), yticklabels=types_crimes)
plt.title('Types de crimes par heure')
plt.xlabel('Heure')
plt.ylabel('Type de crime')

# 10. Distribution des crimes par quartier (pie chart)
plt.subplot(5, 2, 10)
plt.pie(comptes_quartiers, labels=quartiers, autopct='%1.1f%%', startangle=90)
plt.title('Distribution des crimes par quartier')

plt.tight_layout()
plt.savefig('analyse_criminalite_etendue.png')
plt.close()

print("Visualisation sauvegardée sous 'analyse_criminalite_etendue.png'")

spark.stop()