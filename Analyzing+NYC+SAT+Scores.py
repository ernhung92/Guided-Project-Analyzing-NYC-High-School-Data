
# coding: utf-8

# # Project

# This projects is about analyzing New York City's data on high school SAT scores and determining if the U.S. educational system's standardized tests such as the SAT, AP exams are unfair to certain groups in the demographic. 
# 
# So we want to investigate how SAT scores correlate with factors like race, gender, income, and more. 

# # The Data

# Here are all of the files in the folder:
# 
# 1. ap_2010.csv - Data on AP test results
# 2. class_size.csv - Data on class size
# 3. demographics.csv - Data on demographics
# 4. graduation.csv - Data on graduation outcomes
# 5. hs_directory.csv - A directory of high schools
# 6. sat_results.csv - Data on SAT scores
# 7. survey_all.txt - Data on surveys from all schools
# 8. survey_d75.txt - Data on surveys from New York City district 75

# For a convenient way to read and store the data, we'll read them into a dataframe and store all of the dataframes in a dictionary, so we can easily reference them

# In[1]:

import pandas as pd
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}

for file in data_files:
    df = pd.read_csv(file)
    key_name = file.replace(".csv", "")
    data[key_name] = df
data


# # Exploring the SAT Data

# "sat_results" contains the SAT scores for each high school in New York city. Let's explore the dataset to see what we can discover

# In[2]:

print(data["sat_results"].head())


# Observations:
# 1. the DBN appears to be a unique ID for each school: only a single row for each high school, so each DBN is unique in the SAT data.
# 2. Based on the first five rows, we can see that we only have data about high schools.
# 3. We may want to combine the results of the SAT scores to make the scores easier to analyze

# # Exploring the Remaining Data

# In[3]:

data["ap_2010"].head()


# This dataset appears to have a DBN column as well.

# In[4]:

data["class_size"].head()


# The dataset doesn't have a DBN column, but we can combine the CSD and School code together to form a new DBN column. 

# In[5]:

data["demographics"].head()


# In[6]:

data["graduation"].head()


# There duplicated rows in this dataset. So we'll need to do some preprocessing to ensure that each DBN is unique within each data set, otherwise there would be problems merging datasets together

# In[7]:

data["hs_directory"].head()


# dbn column needs to be converted to DBN to be consistent with the rest of the data

# # Reading in the Survey Data

# The survey files are tab delimited and encoded with Windows-1252 encoding. We'll need to specify the encoding to ensure that we can read the data properly. 
# 
# After we read in the survey data, we can combine it into a single dataframe using pd.concat()

# In[8]:

all_survey = pd.read_csv("survey_all.txt",delimiter="\t",encoding="windows-1252")
survey_d75 = pd.read_csv("survey_d75.txt",delimiter="\t",encoding="windows-1252")
survey = pd.concat([all_survey, survey_d75], axis=0)
survey.head()


# Observations:
# 1. There are over 2000 columns, nearly all of which we don't need. We may have to filter the data to remove the unnecessary ones. Removing irrelevant columns helps us to print out the dataframe out and find correlation within it
# 2. It doesn't look like there is a dbn column, but after opening the file in Excel, there's a dbn column. so we need to convert it to uppercase.

# In[9]:

survey["DBN"] = survey["dbn"]
survey_fields = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_11", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]

survey = survey[survey_fields]
data["survey"] = survey
data["survey"].head()


# # Inserting DBN Fields

# From looking at the above dataset, we can tell that the DBN in the "sat_results" data is just a combination of the CSD and SCHOOL CODE columns in the "class_size" data. The main difference is that the DBN is padded, so that the CSD portion of it always consists of two digits. That means we'll need to add a leading 0 to the CSD if the CSD is less than two digits long

# In[10]:

def add_to_single_digits(number):
    string = str(number)
    if len(string) >= 2:
        return string
    else:
        return string.zfill(2) # adds a 0 to the front of the string
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(add_to_single_digits)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]
data["class_size"].head()


# # Combining the SAT Scores

# We want to totals up the SAT scores in order to simplify the process of correlating scores with demographic factors(rather than comparing with 3 different scores)
# 
# Before we do, we need to verify that the scores are under the correct type

# In[11]:

data["sat_results"].dtypes


# We need to convert the 3 scores into numeric
# 
# This can be done with pd.to_numeric() method. We'll pass errors="coerce" so that pandas treats any invalid strings it can't convert to number as missing values instead

# In[12]:

column_names = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in column_names:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][column_names[0]] + data['sat_results'][column_names[1]] + data['sat_results'][column_names[2]]
data['sat_results']['sat_score'].head()


# # Parsing Geographic Coordinates for Schools

# In[13]:

data["hs_directory"]["Location 1"].head(10)


# This column contains the latitude and longitude coordinates for each school. If we want to uncover geographic patterns in this data. We need to extract the latitude and longitude coordinates

# In[14]:

data["hs_directory"]["Location 1"].iloc[0]


# the first value in the parenthesis is the latitude, and the second value is the longitude

# Let's write two functions to extract the longitude and latitude

# In[15]:

import re
def extract_latitude(string):
    # Extracts everything inside the parentheses
    coordinates = re.findall("\(.+\)", string)
    latitude = coordinates[0].split(",")[0].replace("(","")
    return latitude
data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(extract_latitude)
data["hs_directory"]["lat"].head()


# In[16]:

def extract_longitude(string):
    coordinates = re.findall("\(.+\)", string)
    longitude = coordinates[0].split(",")[1].replace(")","")
    return longitude
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(extract_longitude)
data["hs_directory"]["lon"].head()


# In[17]:

data["hs_directory"].dtypes


# The lat and lon columns are still object types. We need to convert them to numeric values

# In[18]:

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"],errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"],errors="coerce")
data["hs_directory"].dtypes


# In[19]:

data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]


# # Condensing the Class Size Data Set

# In[20]:

data["class_size"].head()


# It looks like the first few rows belongs to the same school, which is why the values are duplicated. It looks like each school has unique values for GRADE, PROGRAM TYPE, CORE SUBJECT (MS CORE and 9-12 ONLY) and CORE COURSE (MS CORE and 9-12 ONLY)

# In[21]:

data["class_size"]["GRADE "].value_counts()


# We are only interested in high school students. So we'll probably only look at GRADE == 09-12

# In[22]:

data["class_size"]["PROGRAM TYPE"].value_counts()


# Each school can have multiple program types. Because Gen ED is the largest category by far, we'll use it

# In[23]:

data["class_size"] = data["class_size"][data["class_size"]["GRADE "]=="09-12"]
data["class_size"] = data["class_size"][data["class_size"]["PROGRAM TYPE"]=="GEN ED"]
data["class_size"].head()


# So it looks like we still have duplicated values. This is due to the CORE COURSE (MS CORE and 9-12 ONLY) and CORE SUBJECT (MS CORE and 9-12 ONLY) columns. CORE COURSE (MS CORE and 9-12 ONLY) and CORE SUBJECT (MS CORE and 9-12 ONLY) seem to pertain to different kinds of classes

# In[24]:

data["class_size"]["CORE SUBJECT (MS CORE and 9-12 ONLY)"].value_counts()


# We want to include every single class a school can offer. What we can do is take the average across all of the classes a school offers. This will give us the unique values, while incorporating as much data as possible into the average

# In[25]:

import numpy as np
class_size = data["class_size"]
class_size = class_size.groupby("DBN").agg(np.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size
data["class_size"].head()


# Awesome! Now all rows are unique

# # Condensing the Demographics Data Set

# In[26]:

data["demographics"].head()


# In this case, the only column that prevents a given DBN from being unique is schoolyear. We only want to select rows where schoolyear is "20112012". This will give us the most recent year of data, and also match our SAT results data

# In[27]:

demographics = data["demographics"]
demographics = demographics[demographics["schoolyear"]==20112012]
data["demographics"] = demographics
data["demographics"].head()


# Data rows are unique now

# # Condensing the Graduation Data Set

# In[28]:

data["graduation"].head()


# The Demographic and Cohort columns are what prevent DBN from being unique in the graduation data. A Cohort appears to refer to the year the data represents, and the Demographic appears to refer to a specific demographic group. In this case, we want to pick data from the most recent Cohort available, which is 2006. We also want data from the full cohort, so we'll only pick rows where Demographic is Total Cohort

# In[29]:

graduation = data["graduation"]
graduation = graduation[graduation["Cohort"]=="2006"]
graduation = graduation[graduation["Demographic"]=="Total Cohort"]
data["graduation"] = graduation
data["graduation"].head()


# # Converting AP Test Scores

# In[30]:

data["ap_2010"].head()


# In[31]:

data["ap_2010"].dtypes


# We need to convert the test scores from strings to numeric values. AP scores are interesting because many high school students take AP exams, particularly those who attend academically challenging institutions. 
# 
# It will be interesting to find out whether AP exam scores are correlated with SAT scores across high schools. To determine this, we'll need to convert the AP exam scores in the ap_2010 data set to numeric values first. 

# In[32]:

cols_to_convert = ["AP Test Takers ", "Total Exams Taken", "Number of Exams with scores 3 4 or 5"]
for c in cols_to_convert:
    data["ap_2010"][c] = pd.to_numeric(data["ap_2010"][c],errors="coerce")
data["ap_2010"].dtypes


# # Combining Our Datasets

# In[33]:

combined = data["sat_results"]
combined = combined.merge(data["ap_2010"],on="DBN",how="left")
combined = combined.merge(data["graduation"],on="DBN",how="left")
combined = combined.merge(data["class_size"], on="DBN", how="inner")
combined = combined.merge(data["demographics"], on="DBN", how="inner")
combined = combined.merge(data["survey"], on="DBN", how="inner")
combined = combined.merge(data["hs_directory"], on="DBN", how="inner")
combined.head()


# # Filling In Missing Values

# In[34]:

combined.isnull().sum()


# There are a lot of columns with missing values.
# 
# How do we deal with them? 
# 1. We can delete the corresponding rows and risk losing information in the process
# 2. We can fill in the missing values with the corresponding average of the column

# In[35]:

means = combined.mean()
combined = combined.fillna(means)
combined = combined.fillna(0)
combined.isnull().sum()


# # Adding a School District Column for Mapping

# The school district is just the first two characters of the DBN. We can apply a function over the DBN column of combined that pulls out the first two letters

# In[36]:

def get_first_two_charcs(dbn_string):
    return dbn_string[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_charcs)
combined.head()


# # Finding Correlations With the r Value

# Correlations tell us how closely related two columns are. We'll be using the r value, also called Pearson's correlation coefficient, which measures how closely two sequences of numbers are correlated.
# 
# An r value falls between -1 and 1. The value tells us whether two columns are positively correlated, not correlated, or negatively correlated. The closer to 1 the r value is, the stronger the positive correlation between the two columns. The closer to -1 the r value is, the stronger the negative correlation. The closer to 0, the weaker the correlation

# In general, r values above .25 or below -.25 are enough to qualify a correlation as interesting. An r value isn't perfect, and doesn't indicate that there's a correlation -- just the possiblity of one. To really assess whether or not a correlation exists, we need to look at the data using a scatterplot to see its "shape."

# In[37]:

correlations = combined.corr()
correlations = correlations["sat_score"]
correlations


# Unsurprisingly, SAT Critical Reading Avg. Score, SAT Math Avg. Score, SAT Writing Avg. Score, and sat_score are strongly correlated with sat_score.
# 
# We can also make some other observations:
# 
# 1. total_enrollment has a strong positive correlation with sat_score. This is surprising because we'd expect smaller schools where students receive more attention to have higher scores. However, it looks like the opposite is true -- larger schools tend to do better on the SAT.
# 
# 2. Other columns that are proxies for enrollment correlate similarly. These include total_students, N_s, N_p, N_t, AP Test Takers, Total Exams Taken, and NUMBER OF SECTIONS.
# 
# 3. Both the percentage of females (female_per) and number of females (female_num) at a school correlate positively with SAT score, whereas the percentage of males (male_per) and the number of males (male_num) correlate negatively. This could indicate that women do better on the SAT than men.
# 
# 4. Teacher and student ratings of school safety (saf_t_11, and saf_s_11) correlate with sat_score.
# 
# 5. Student ratings of school academic standards (aca_s_11) correlate with sat_score, but this does not hold for ratings from teachers and parents (aca_p_11 and aca_t_11).
# 
# 6. There is significant racial inequality in SAT scores (white_per, asian_per, black_per, hispanic_per).
# 
# 7. The percentage of English language learners at the school (ell_percent, frl_percent) has a strong negative correlation with SAT scores.

# # Plotting Enrollment vs. SAT Scores

# In[38]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.scatter(combined["total_enrollment"], combined["sat_score"])
plt.show()


# Even though there was an indication of a strong correlation between these two columns, the graph doesn't back it up. If they were, all the points would have lined up. Instead we have a large cluster of schools and a few others going off in 3 different directions
# 
# However, there is an interesting cluster of points at the bottom left where enrollment and sat_score are both low. This cluster may be what's making the r value so high. 
# 
# Let's extract the names of the schools so we can research them further:

# In[39]:

low_enrollment = combined[combined["total_enrollment"] < 1000]
low_enrollment = low_enrollment[low_enrollment["sat_score"] < 1000]
print(low_enrollment["School Name"])


# It looks like most of the high schools with low total enrollment and low SAT scores have high percentages of English language learners. So maybe it's actually ell_percent that correlates strongly with sat_score, rather than total_enrollment

# In[40]:

combined.plot.scatter("ell_percent","sat_score")
plt.show()


# It looks like ell_percent correlates with sat_score more strongly, because the scatterplot is more linear. However, there's still the cluster of schools that have very high ell_percent values and low sat_score values. This cluster represents the same group of international high schools we investigated earlier.
# 
# In order to explore this relationship, we'll want to map out ell_percent by school district. The map will show us which areas of the city have a lot of English language learners

# # Mapping the Schools With Basemap

# In[41]:

from mpl_toolkits.basemap import Basemap

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = combined["lon"].tolist()
latitudes = combined["lat"].tolist()
m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True)
plt.show()


# From the map, we can see that the school density is highest in Manhattan, and lower in Brooklyn, Bronx, Queens and Staten Island

# Let's shade the percentage of English language learned by area. Indicating green points for low numbers and yellow points for high numbers

# In[42]:

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = combined["lon"].tolist()
latitudes = combined["lat"].tolist()
m.scatter(longitudes, latitudes, s=20, zorder=2, latlon=True, c=combined["ell_percent"], cmap="summer")
plt.show()


# Unfortunately, due to the number of schools, it's hard to interpret the map we made. It looks like uptown Manhattan and parts of Queens have a higher ell_percent. But hard to say. 

# # Calculating District-Level Statistics

# In[43]:

districts = combined.groupby("school_dist").agg(np.mean)
districts.reset_index(inplace=True)
districts.head()


# Now that we have taken the means of all of the columns, we can plot out ell_percent by district. Not only, did we find the mean of ell_percent, by we also took the means of the lon and lat columns, which will give us the coordinates for the center of each district

# In[44]:

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = districts["lon"].tolist()
latitudes = districts["lat"].tolist()
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=districts["ell_percent"], cmap="summer")
plt.show()


# # Plotting Survey Correlations

# In[45]:

# Let's plot the correlations without the unique identifier
survey_fields.remove("DBN")

combined.corr()["sat_score"][survey_fields].plot.bar()


# Observations:
# 1. There are high correlations between N_s(number of student respondents), N_t(number of teacher respondets), N_p(number of parent respondents) and sat_score. Since these columns are correlated with total_enrollment, it makes sense that they would be high.
# 2. It is more interesting that rr_s(the student response rate), or the percentage of students that completed the survey, correlates with sat_score. This might make sense because students who are more likely to fill out surveys may be more likely to also be doing well academically.
# 3. How students and teachers percieved safety (saf_t_11 and saf_s_11) correlate with sat_score. This make sense, as it's hard to teach or learn in an unsafe environment.
# 4. The last interesting correlation is the aca_s_11, which indicates how the student perceives academic standards, correlates with sat_score

# # Exploring Safety and SAT Scores

# In[47]:

# Let's investigate safety scores 
combined.plot.scatter("saf_s_11","sat_score")


# There appears to be a correlation between SAT scores and safety, although it isn't that strong. It looks like there are a few schools with extremely high SAT scores and high safety scores. There are a few schools with low safety scores and low SAT scores. No school with a safety score lower than 6.5 has an average SAT score higher than 1500 or so.

# # Plotting Safety by District

# In[48]:

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = districts["lon"].tolist()
latitudes = districts["lat"].tolist()
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=districts["saf_s_11"], cmap="summer")
plt.show()


# It looks like Upper Manhattan and parts of Queens and the Bronx tend to have lower safety scores, whereas Brooklyn has high safety scores.

# # Racial Differences in SAT Scores

# In[50]:

races = ["white_per", "asian_per", "black_per", "hispanic_per"]
combined.corr()["sat_score"][races].plot.bar()


# It looks like a higher percentage of white or asian students at a school correlates positively with sat score, whereas a higher percentage of black or hispanic students correlates negatively with sat score

# In[51]:

# Let's explore SAT scores of schools by the percent of Hispanics
combined.plot.scatter("hispanic_per", "sat_score")


# In[52]:

# Let's look for schools with hispanic population of 95%
print(combined[combined["hispanic_per"] > 95]["SCHOOL NAME"])


# The schools listed above appear to primarily be geared towards recent immigrants to the US. These schools have a lot of students who are learning English, which would explain the lower SAT scores.

# What about schools with percentage of hispanics < 10% and SAT score > 1800?

# In[54]:

print(combined[(combined["hispanic_per"] < 10) & (combined["sat_score"] > 1800)]["SCHOOL NAME"])


# Many of the schools above appear to be specialized science and technology schools that receive extra funding, and only admit students who pass an entrance exam. This doesn't explain the low hispanic_per, but it does explain why their students tend to do better on the SAT -- they are students from all over New York City who did well on a standardized test.

# # Gender Differences in SAT Score

# In[55]:

genders = ["male_per", "female_per"]
combined.corr()["sat_score"][genders].plot.bar()


# we can see that a high percentage of females at a school positively correlates with SAT score, whereas a high percentage of males at a school negatively correlates with SAT score. Though neither correlation is extremely strong.

# In[56]:

# Let's explore female SAT scores
combined.plot.scatter("female_per", "sat_score")


# there doesn't seem to be any real correlation between sat_score and female_per. However, there is a cluster of schools with a high percentage of females (60 to 80), and high SAT scores.

# In[57]:

print(combined[(combined["female_per"] > 60) & (combined["sat_score"] > 1700)]["SCHOOL NAME"])


# These schools appears to be very selective liberal arts schools that have high academic standards.
