{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "bd43aceb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn import datasets\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f40a0ee1",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'Table_1 (1).csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-11-7ce99f39def8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mattrdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Table_1 (1).csv\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36mread_csv\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)\u001b[0m\n\u001b[0;32m    608\u001b[0m     \u001b[0mkwds\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkwds_defaults\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    609\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 610\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    611\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    612\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    460\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    461\u001b[0m     \u001b[1;31m# Create the parser.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 462\u001b[1;33m     \u001b[0mparser\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    463\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    464\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m    817\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    818\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 819\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    820\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    821\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[1;34m(self, engine)\u001b[0m\n\u001b[0;32m   1048\u001b[0m             )\n\u001b[0;32m   1049\u001b[0m         \u001b[1;31m# error: Too many arguments for \"ParserBase\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1050\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mmapping\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# type: ignore[call-arg]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1051\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1052\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_failover_to_python\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, src, **kwds)\u001b[0m\n\u001b[0;32m   1865\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1866\u001b[0m         \u001b[1;31m# open handles\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1867\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_open_handles\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1868\u001b[0m         \u001b[1;32massert\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhandles\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1869\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;32min\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;34m\"storage_options\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"encoding\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"memory_map\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"compression\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_open_handles\u001b[1;34m(self, src, kwds)\u001b[0m\n\u001b[0;32m   1360\u001b[0m         \u001b[0mLet\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mreaders\u001b[0m \u001b[0mopen\u001b[0m \u001b[0mIOHanldes\u001b[0m \u001b[0mafter\u001b[0m \u001b[0mthey\u001b[0m \u001b[0mare\u001b[0m \u001b[0mdone\u001b[0m \u001b[1;32mwith\u001b[0m \u001b[0mtheir\u001b[0m \u001b[0mpotential\u001b[0m \u001b[0mraises\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1361\u001b[0m         \"\"\"\n\u001b[1;32m-> 1362\u001b[1;33m         self.handles = get_handle(\n\u001b[0m\u001b[0;32m   1363\u001b[0m             \u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1364\u001b[0m             \u001b[1;34m\"r\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\io\\common.py\u001b[0m in \u001b[0;36mget_handle\u001b[1;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[0;32m    640\u001b[0m                 \u001b[0merrors\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"replace\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    641\u001b[0m             \u001b[1;31m# Encoding\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 642\u001b[1;33m             handle = open(\n\u001b[0m\u001b[0;32m    643\u001b[0m                 \u001b[0mhandle\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    644\u001b[0m                 \u001b[0mioargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'Table_1 (1).csv'"
     ]
    }
   ],
   "source": [
    "attrdata = pd.read_csv(\"Table_1 (1).csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e789646",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a717376b",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata.drop(0,inplace=True)\n",
    "attrdata.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23e92cb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata.dropna(axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4c226b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc62dde7",
   "metadata": {},
   "outputs": [],
   "source": [
    "gender_dict = attrdata[\"Gender \"].value_counts()\n",
    "gender_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a32939f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata['Gender '].value_counts().plot(kind='bar',color=['salmon','lightblue'],title=\"Count of different gender\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "623a1bdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a plot for crosstab\n",
    "pd.crosstab(attrdata['Gender '],attrdata['Stay/Left']).plot(kind=\"bar\",figsize=(10,6))\n",
    "plt.title(\"Stay/Left vs Gender\")\n",
    "plt.xlabel(\"Stay/Left\")\n",
    "plt.ylabel(\"No of people who left based on gender\")\n",
    "plt.legend([\"Left\",\"Stay\"])\n",
    "plt.xticks(rotation=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a303bff",
   "metadata": {},
   "outputs": [],
   "source": [
    "promoted_dict = attrdata[\"Promoted/Non Promoted\"].value_counts()\n",
    "promoted_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad6b5297",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata['Promoted/Non Promoted'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title=\"Promoted and Non Promoted\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1c366b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a plot for crosstab\n",
    "\n",
    "pd.crosstab(attrdata['Promoted/Non Promoted'],attrdata['Stay/Left']).plot(kind=\"bar\",figsize=(10,6))\n",
    "plt.title(\"Stay/Left vs Promoted/Non Promoted\")\n",
    "plt.xlabel(\"Stay/Left\")\n",
    "plt.ylabel(\"No. of people who left/stay based on promotion\")\n",
    "plt.legend([\"Left\",\"Stay\"])\n",
    "plt.xticks(rotation=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5447ed21",
   "metadata": {},
   "outputs": [],
   "source": [
    "func_dict = attrdata[\"Function\"].value_counts()\n",
    "func_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e4ac0f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata['Function'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title=\"Functions in organization\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "470c75e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a plot for crosstab\n",
    "\n",
    "pd.crosstab(attrdata['Function'],attrdata['Stay/Left']).plot(kind=\"bar\",figsize=(10,6))\n",
    "plt.title(\"Stay/Left vs Function\")\n",
    "plt.xlabel(\"Stay/Left\")\n",
    "plt.ylabel(\"No. of people who left/stay based on function of organization\")\n",
    "plt.legend([\"Left\",\"Stay\"])\n",
    "plt.xticks(rotation=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7618090",
   "metadata": {},
   "outputs": [],
   "source": [
    "Hiring_dict = attrdata[\"Hiring Source\"].value_counts()\n",
    "Hiring_dictss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a8547b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "Marital_dict = attrdata[\"Marital Status\"].value_counts()\n",
    "print(Marital_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d94e9a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "Emp_dict = attrdata[\"Emp. Group\"].value_counts()\n",
    "Emp_dict['other group'] = 1\n",
    "print(Emp_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "565dd0d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "job_dict = attrdata[\"Job Role Match\"].value_counts()\n",
    "job_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ed3b49f",
   "metadata": {},
   "outputs": [],
   "source": [
    "attrdata['Job Role Match'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title=\"Job Role Match\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b771a6f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a plot for crosstab\n",
    "\n",
    "pd.crosstab(attrdata['Job Role Match'],attrdata['Stay/Left']).plot(kind=\"bar\",figsize=(10,6))\n",
    "plt.title(\"Stay/Left vs Job Role Match\")\n",
    "plt.xlabel(\"Stay/Left\")\n",
    "plt.legend([\"Left\",\"Stay\"])\n",
    "plt.xticks(rotation=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3572a1c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "tenure_dict = attrdata[\"Tenure Grp.\"].value_counts()\n",
    "print(tenure_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe439c46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Its Age vs stay/left\n",
    "sns.jointplot(x='Stay/Left',y='Age in YY.',data=attrdata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "114972f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.jointplot(x='Stay/Left',y='Experience (YY.MM)',data=attrdata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e141f420",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's make our correlation matrix visual\n",
    "corr_matrix=attrdata.corr()\n",
    "fig,ax=plt.subplots(figsize=(15,10))\n",
    "ax=sns.heatmap(corr_matrix,\n",
    "               annot=True,\n",
    "               linewidths=0.5,\n",
    "               fmt=\".2f\"\n",
    "              )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7cc75a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "location_dict = attrdata[\"Location\"].value_counts()\n",
    "print(location_dict)\n",
    "\n",
    "location_dict_new = {\n",
    "    'Chennai':       7,\n",
    "    'Noida':         6,\n",
    "    'Bangalore':     5,\n",
    "    'Hyderabad':     4,\n",
    "    'Pune':          3,\n",
    "    'Madurai':       2,\n",
    "    'Lucknow':       1,\n",
    "    'other place':   0,\n",
    "}\n",
    "\n",
    "print(location_dict_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0dde3071",
   "metadata": {},
   "outputs": [],
   "source": [
    "def location(x):\n",
    "    if str(x) in location_dict_new.keys():\n",
    "        return location_dict_new[str(x)]\n",
    "    else:\n",
    "        return location_dict_new['other place']\n",
    "    \n",
    "data_l = attrdata[\"Location\"].apply(location)\n",
    "attrdata['New Location'] = data_l\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b620a92b",
   "metadata": {},
   "outputs": [],
   "source": [
    "gen = pd.get_dummies(attrdata[\"Function\"])\n",
    "gen.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f4ea1e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "hr = pd.get_dummies(attrdata[\"Hiring Source\"])\n",
    "hr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5762d5cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Mar(x):\n",
    "    if str(x) in Marital_dict.keys() and Marital_dict[str(x)] > 100:\n",
    "        return str(x)\n",
    "    else:\n",
    "        return 'other status'\n",
    "    \n",
    "data_l = attrdata[\"Marital Status\"].apply(Mar)\n",
    "attrdata['New Marital'] = data_l\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48034822",
   "metadata": {},
   "outputs": [],
   "source": [
    "Mr = pd.get_dummies(attrdata[\"New Marital\"])\n",
    "Mr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b642593",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Promoted(x):\n",
    "    if x == 'Promoted':\n",
    "        return int(1)\n",
    "    else:\n",
    "        return int(0)\n",
    "\n",
    "data_l = attrdata[\"Promoted/Non Promoted\"].apply(Promoted)\n",
    "attrdata['New Promotion'] = data_l\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "765de13e",
   "metadata": {},
   "outputs": [],
   "source": [
    "Emp_dict_new = {\n",
    "    'B1': 4,\n",
    "    'B2': 3,\n",
    "    'B3': 2,\n",
    "    'other group': 1,\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "638140e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def emp(x):\n",
    "    if str(x) in Emp_dict_new.keys():\n",
    "        return str(x)\n",
    "    else:\n",
    "        return 'other group'\n",
    " \n",
    "data_l = attrdata[\"Emp. Group\"].apply(emp)\n",
    "attrdata['New EMP'] = data_l\n",
    "\n",
    "emp = pd.get_dummies(attrdata[\"New EMP\"])\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c442e9ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Job(x):\n",
    "    if x == 'Yes':\n",
    "        return int(1)\n",
    "    else:\n",
    "        return int(0)\n",
    "    \n",
    "data_l = attrdata[\"Job Role Match\"].apply(Job)\n",
    "attrdata['New Job Role Match'] = data_l\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56546641",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Gen(x):\n",
    "    if x in gender_dict.keys():\n",
    "        return str(x)\n",
    "    else:\n",
    "        return 'other'\n",
    "\n",
    "data_l = attrdata[\"Gender \"].apply(Gen)\n",
    "attrdata['New Gender'] = data_l\n",
    "attrdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f0966f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "gend = pd.get_dummies(attrdata[\"New Gender\"])\n",
    "gend.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ba7b024",
   "metadata": {},
   "outputs": [],
   "source": [
    "tengrp = pd.get_dummies(attrdata[\"Tenure Grp.\"])\n",
    "tengrp.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1be2503",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.concat([attrdata, hr, Mr, emp, tengrp, gen, gend], axis = 1)\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca9639b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "894e919b",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.drop([\"table id\", \"name\", \"Marital Status\",\"Promoted/Non Promoted\",\"Function\",\"Emp. Group\",\"Job Role Match\",\"Location\"\n",
    "              ,\"Hiring Source\",\"Gender \", 'Tenure', 'New Gender', 'New Marital', 'New EMP'],axis=1,inplace=True)\n",
    "\n",
    "dataset1 = dataset.drop(['Tenure Grp.', 'phone number'], axis = 1)\n",
    "dataset1.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "593f0b6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's make our correlation matrix visual\n",
    "corr_matrix=dataset1.corr()\n",
    "fig,ax=plt.subplots(figsize=(15,10))\n",
    "ax=sns.heatmap(corr_matrix,\n",
    "               annot=True,\n",
    "               linewidths=0.5,\n",
    "               fmt=\".2f\"\n",
    "              )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cae2bd4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target \n",
    "\"\"\"\n",
    "def Target(x):\n",
    "    if x in \"Stay\":\n",
    "        return False\n",
    "    else:\n",
    "        return True\n",
    "    \n",
    "data_l = dataset1[\"Stay/Left\"].apply(Target)\n",
    "dataset1['Stay/Left'] = data_l\n",
    "\"\"\"\n",
    "dataset1['Stay/Left'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d79268a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset1.to_csv(\"processed table.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16254bb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv(\"processed table.csv\")\n",
    "dataset = pd.DataFrame(dataset)\n",
    "y = dataset[\"Stay/Left\"]\n",
    "X = dataset.drop(\"Stay/Left\",axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4c12625",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)\n",
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dca0a9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn import svm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ea78a9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr=LogisticRegression(C = 0.1, random_state = 42, solver = 'liblinear')\n",
    "dt=DecisionTreeClassifier()\n",
    "rm=RandomForestClassifier()\n",
    "gnb=GaussianNB()\n",
    "knn = KNeighborsClassifier(n_neighbors=3)\n",
    "svm = svm.SVC(kernel='linear')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a522d2b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "for a,b in zip([lr,dt,knn,svm,rm,gnb],[\"Logistic Regression\",\"Decision Tree\",\"KNN\",\"SVM\",\"Random Forest\",\"Naive Bayes\"]):\n",
    "    a.fit(X_train,y_train)\n",
    "    prediction=a.predict(X_train)\n",
    "    y_pred=a.predict(X_test)\n",
    "    score1=accuracy_score(y_train,prediction)\n",
    "    score=accuracy_score(y_test,y_pred)\n",
    "    msg1=\"[%s] training data accuracy is : %f\" % (b,score1)\n",
    "    msg2=\"[%s] test data accuracy is : %f\" % (b,score)\n",
    "    print(msg1)\n",
    "    print(msg2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61e6f48a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_scores={'Logistic Regression':lr.score(X_test,y_test),\n",
    "             'KNN classifier':knn.score(X_test,y_test),\n",
    "             'Support Vector Machine':svm.score(X_test,y_test),\n",
    "             'Random forest':rm.score(X_test,y_test),\n",
    "              'Decision tree':dt.score(X_test,y_test),\n",
    "              'Naive Bayes':gnb.score(X_test,y_test)\n",
    "             }\n",
    "model_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca9abf0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "rm_y_preds = rm.predict(X_test)\n",
    "\n",
    "print(classification_report(y_test,rm_y_preds))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd4c8dae",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "lr_y_preds = lr.predict(X_test)\n",
    "\n",
    "print(classification_report(y_test,lr_y_preds))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a9fc7ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_compare=pd.DataFrame(model_scores,index=['accuracy'])\n",
    "model_compare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a09b9fa0",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_compare.T.plot(kind='bar') # (T is here for transpose)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5490473f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic regression\n",
    "feature_dict=dict(zip(dataset.columns,list(lr.coef_[0])))\n",
    "feature_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5eb77b01",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_df=pd.DataFrame(feature_dict,index=[0])\n",
    "feature_df.T.plot(kind=\"bar\",legend=False,title=\"Feature Importance\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2f87dc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "# Save the trained model as a pickle string.\n",
    "saved_model = pickle.dumps(lr)\n",
    "\n",
    "# Load the pickled model\n",
    "lr_from_pickle = pickle.loads(saved_model)\n",
    "\n",
    "# Use the loaded pickled model to make predictions\n",
    "lr_from_pickle.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5637fa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# loading dependency\n",
    "import joblib\n",
    "\n",
    "# saving our model - model - model , filename - model_lr\n",
    "joblib.dump(lr , 'model_lr')\n",
    "\n",
    "# opening the file- model_jlib\n",
    "m_jlib = joblib.load('model_lr')\n",
    "\n",
    "# check prediction\n",
    "m_jlib.predict(X_test) # similar output"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
