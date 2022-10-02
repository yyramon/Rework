import PyUber
import json
import sys, os, glob
import pandas as  pd
from datetime import datetime
conn = PyUber.connect(datasource="F28_PROD_XEUS")
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import numpy as np
import matplotlib.pyplot as plt



catConfig=pd.read_csv(r'\\isSHFS.intel.com\isAnalysis$\1274_MAODATA\Config\LITHO\NIKON\WET\Layers\RWK_NEW\RWK_CAT.csv', index_col=False).rename(columns={'CATEGORY':'REWORK_CATEGORY'})
weeksBack = 50


entityDataQuery="""
SELECT DISTINCT e.FACILITY, e.ENTITY
FROM   F_ENTITY e
WHERE  e.cluster_name='ASMLWET' or e.cluster_name='OVLN'
"""

query = """select * from (SELECT lot.LOT as LOT,
lot.LOT_TYPE,
lot.lot_process,
lot.ENTITY,
LEAD(lot.ENTITY) OVER (PARTITION BY lot.lot ORDER BY lot.lot) as RegEntity,
substr(lot.RETICLE,7,3) as layer,
lot.RETICLE,
lot.OUT_WW,
lot.OUT_WAFER_QTY,
lot.OPERATION,
lot.OPER_SHORT_DESC,
lot.OPERATION||' '||lot.OPER_SHORT_DESC as OP_LAYER,
LEAD(lot.OPERATION) OVER (PARTITION BY lot.lot ORDER BY lot.lot) as REG_OPERATION,
lot.OUT_DATE,
lot.PRODUCT,
substr(lot.product,0,6) as product_family,
lot.REWORK_FLAG,
lot.HOLD_FLAG,
lot.LAST_HOLD_COMMENT,
lot.HOTLOT_AT_OPER, lot.ON_HOLD_AT_OPER, lot.INSTRUCTION_FLAG
              , lot.DISPO_FLAG, lot.REWORK_CATEGORY, lot.LOT_ACCUM_REWORK_COUNTER, lot.LAYER_REWORK_COUNTER,
     lot.OUT_SHIFT
FROM   F_LOT_FLOW lot
where
(  lot.entity like 'SA%' or lot.ENTITY like 'OM%' )
--(lot.entity like 'SAU%' or lot.entity like 'SAQ%' or lot.entity like 'SAS%' or lot.entity like 'SAR%' or lot.entity like 'OM%')
--and lot.lot in ('N1084780','N1078460','N1045260')
and lot.lot_type in ('PROD','ENG')
and lot.out_wafer_qty is not null
and lot.lot_process like '1274%'
and lot.out_date>=Trunc((current date) - (dayofweek(current date)-1) days)-7*{wb}
order by  lot.lot ASC, lot.OUT_DATE ASC)
where reticle is not null

""".format(wb=weeksBack)
# #
# #
# #
def updateData():
    entDf = pd.read_sql(entityDataQuery, conn)
    entity = tuple(sorted(entDf.ENTITY.unique().tolist()))
    # entity=''
    df = pd.read_sql(query, conn)
    df = pd.merge(entDf, df, on='ENTITY')
    df = pd.merge(df, layerConfig, on='LAYER')
    df = pd.concat([temp, df]).reset_index(drop=True)

    df['REWORK_CATEGORY'] = df['REWORK_CATEGORY'].map(catConfig.set_index('REWORK_CATEGORY')['DESC'])
    df['LAST_UPDATED'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    df_temp = df.astype('str')
    df_temp = df_temp.drop_duplicates(subset=None, keep="last")
    df_temp.to_csv(os.path.join(path, 'RW_WET.csv'), index=False)
    # df_temp.to_csv(os.path.join(path, 'RW_WET.csv.gz'), index=False, compression='gzip')
    print('RW file saved ' + datetime.now().strftime('%H:%M:%S'))
    return df
# #
current_ww=datetime.now().strftime('%Y')+datetime.now().strftime('%U')
source=('F28_PROD_XEUS','F32_PROD_XEUS','D1D_PROD_XEUS')
path=r"\\ISSHFS.intel.com\ISanalysis$\1274_MAODATA\Config\LITHO\ASML_Scanner\CT\Rework_forML"
layerConfig=pd.read_csv(r"\\ISSHFS.intel.com\ISanalysis$\1274_MAODATA\Config\LITHO\ASML_Scanner\CT\LineDown_wetDry\layerOrderAll.csv", index_col=False)
layerConfig['segment'].replace({"FE":"1-FE","SSAFI0":"2-SSAFI0","SSAFI1":"3-SSAFI1","PD":"4-PD","BE":"5-BE"}, inplace=True)

temp=pd.DataFrame()

rwk_db=updateData()

# ###### data review REWORK
reg_lot_path = r"\\ISSHFS.intel.com\ISanalysis$\1274_MAODATA\Litho\ASML\1274_ASML_Dashboard\REG lot level"
files = [fn for fn in glob.glob(reg_lot_path + '\*.csv') if not os.path.basename(fn).startswith('All')]

li = []

for file in files:
    try:
        df_all = pd.read_csv(file,index_col=None,usecols=['LOT','REG_OPERATION','OVER_LAYER', 'UNDER_LAYER', 'OVERLAYER_OPERATION','OVERLAYER_ENTITY', 'OL_START_CHK','UNDERLAYER_OPERATION','UNDERLAYER_ENTITY','PERCENT_MEASURED_MIN'])
        li.append(df_all)
        print(file)
    except OSError:
                pass

df_all_gen = pd.concat(li,axis=0,ignore_index=True)


df_RWK = df_all_gen[['LOT', 'REG_OPERATION','OVERLAYER_ENTITY','UNDERLAYER_ENTITY']]
df_RWK['OL_UL_TOOLS'] = df_RWK['OVERLAYER_ENTITY'] + '_' + df_RWK['UNDERLAYER_ENTITY']
df_RWK = df_RWK[~df_RWK['OL_UL_TOOLS'].isnull()]
df_RWK  = df_RWK[~(df_RWK['OL_UL_TOOLS'].str.contains('SAR|SAV|SAS')==1)]

# rwk_db = rwk_db.mask(rwk_db.eq('None')).dropna()
#rwk_db = rwk_db.astype({'REG_OPERATION': int})

rwk_db['REG_OPERATION'] = rwk_db['REG_OPERATION'].astype('float')

test = df_RWK.merge(rwk_db, on=['LOT','REG_OPERATION'])

del df_RWK
del df_all_gen
del rwk_db

y = test.REWORK_FLAG
y = y.map({'N':0,'Y':1}).values.ravel()
X = pd.get_dummies(pd.Series(test.OL_UL_TOOLS.to_list()))

clf_xgBoost = xgb.XGBClassifier()

param_grid = {
    'max_depth':[2,4,6,8],
    'n_estimators':[20, 50, 75]
}

gs_cv = GridSearchCV(clf_xgBoost,param_grid, cv=5)

gs_cv.fit(X, y)
# clf_xgBoost = xgb.XGBClassifier(
#     max_depth = 4,
#     subsample = 0.8,
#     colsample_bytree = 0.7,
#     colsample_bylevel = 0.7,
#     scale_pos_weight = 9,
#     min_child_weight = 0,
#     reg_alpha = 4,
#     n_jobs = 4,
#     objective = 'binary:logistic'
# )
# Fit the models


print('best parameters are: ', gs_cv.best_params_,'\n'
      'best score are : ',gs_cv.best_score_)


clf_xgBoost = xgb.XGBClassifier(max_depth = 2, n_estimators= 20)
clf_xgBoost.fit(X,y)
importance_dict = {}
for import_type in ['weight', 'gain', 'cover']:
    importance_dict['xgBoost-' + import_type] = clf_xgBoost.get_booster().get_score(importance_type=import_type)

#MinMax scale all importances
importance_df = pd.DataFrame(importance_dict).fillna(0)
importance_df = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(importance_df),
    columns=importance_df.columns,
    index=importance_df.index
)

# Create mean column
importance_df['mean'] = importance_df.mean(axis=1)

# Plot the feature importances
importance_df.sort_values('mean').plot(title='30 days data based', kind='bar', figsize=(20, 7))



#
# # xeus_conn = PyUber.connect(datasource="F28_PROD_XEUS")
# # sv_conn = PyUber.connect(datasource="F28_PROD_SCANVIEW")
# #
# # fdc_query  =   """SELECT
# #         fre.run_start_time as fdc_time
# #           ,fsv.variable AS variable
# #           ,fsv.value AS value
# #           ,fre.entity AS polish_tool
# #          ,frl.operation AS polish_operation
# #          ,frl.lot AS lot
# #
# # FROM
# # P_FDC_RUN_ENTITY fre
# # INNER JOIN P_FDC_RUN_LOT frl ON frl.tool_run_id = fre.tool_run_id
# # INNER JOIN P_FDC_RUN_WAFER frw ON frw.tool_run_id = fre.tool_run_id
# # AND frw.tool_run_id = frl.tool_run_id AND frw.lot = frl.lot
# # INNER JOIN P_FDC_SUMMARY_VALUE fsv ON fsv.tool_run_id = fre.tool_run_id
# # AND fsv.tool_run_id = frw.tool_run_id AND (fsv.material_key_context is NULL OR frw.Material_Key_Context = fsv.Material_Key_Context)
# # AND fsv.tool_run_id = frl.tool_run_id
# # WHERE
# #               fre.entity Like 'PLI%'
# #  AND      frl.operation In ('193576'
# # ,'122174')
# #  AND      fsv.variable Like 'Pad%'
# #  AND    (  fre.run_start_time > SYSDATE - 18
# #  and      fre.run_start_time <= SYSDATE - 14 ) """
# #
# # ega_query = """SELECT
# #           a4.value AS ega_value
# #          ,a4.parameter_name AS raw_parameter_name
# #          ,a0.lot AS lot
# #        --  ,a7.wafer AS cr_wafer
# #          ,a1.entity AS nikon_scanner
# #          ,To_Char(a0.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS ega_time
# # FROM
# # P_SPC_ENTITY a1
# # LEFT JOIN P_SPC_Lot a0 ON a0.spcs_id = a1.spcs_id
# # INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a1.spcs_id
# # INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
# # LEFT JOIN P_SPC_CHARTPOINT_MEASUREMENT a7 ON a7.spcs_id = a3.spcs_id and a7.measurement_set_name = a3.measurement_set_name
# # LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
# # AND a4.spcs_id = a7.spcs_id AND a4.measurement_id = a7.measurement_id
# # WHERE  (a0.DATA_COLLECTION_TIME > sysdate - 15
# #        and a0.DATA_COLLECTION_TIME <= sysdate - 12)
# #             and  a4.OPERATION in ('176457')
# #               and a1.entity Like 'SNE%'
# #               and  a4.parameter_name like 'EG%'"""
# #
# # sv_query = """
# #
# # SELECT
# #           sbd.tool_name AS asml_scanner,
# #           sbd.batch_start_date align_time
# #           ,sld.operation as asml_operation
# #          ,sld.lot_number AS lot
# #      --    ,swd.lse_wafer_number AS lse_wafer_number
# #          ,wafd.fine_wafer_quality_x_avg AS fine_wafer_quality_x_avg
# #          ,wafd.fine_wafer_quality_x_stdev AS fine_wafer_quality_x_stdev
# #          ,wafd.fine_wafer_quality_y_avg AS fine_wafer_quality_y_avg
# #          ,wafd.fine_wafer_quality_y_stdev AS fine_wafer_quality_y_stdev
# #          ,wafd.nr_rejected_marks AS nr_rejected_marks
# # FROM
# #      SCANV_SAU_1274.BATCH_DATA sbd
# #     ,SCANV_SAU_1274.LOT_DATA sld
# #     ,SCANV_SAU_1274.WAFER_DATA swd
# #     ,SCANV_SAU_1274.WAFER_ALIGN_FINE_DATA wafd
# # WHERE
# #       sbd.tool_name Like 'SA%'
# #       and sld.operation in ('193917','193922')
# #       AND (sbd.batch_start_date <= SYSDATE - 1
# #       and sbd.batch_start_date > SYSDATE - 8)
# # AND              sbd.deleted=0
# #  AND      sld.batch_idx = sbd.batch_idx
# #  AND      sld.deleted=0
# #  AND      swd.deleted=0
# #  AND      swd.batch_idx = sbd.batch_idx
# #  AND      swd.lot_idx = sld.lot_idx
# #  AND      swd.wafer_idx = wafd.wafer_idx (+)
# #  AND      wafd.deleted (+) = 0
# # """
# #
# # # Pulling Pad life data (FDC)
# # fdcDF = pd.read_sql(fdc_query, xeus_conn)
# #
# # # Pulling fine alignment data for Nikon
# # EGA_df = pd.read_sql(ega_query, xeus_conn)
# #
# # # Fine alignment - rejected and
# # fine_align_df = pd.read_sql(sv_query, sv_conn)
# #
# # #####  PCA for Physical features of Etch and Polish
# #
# # test = fdcDF.merge(EGA_df,left_on=['LOT'],right_on=['LOT'])
# #
# # test = test.merge(fine_align_df,left_on=['LOT'],right_on=['LOT'])
#
# # Phys_df = fdcDF.groupby(['RUN_START_TIME', 'LOT', 'ENTITY','VARIABLE']).mean().unstack()
# #
# # Phys_df = Phys_df.dropna(axis=1, how='all')
# #
# # Phys_df = Phys_df.droplevel(0,axis=1)
# #
# # Phys_df.rename(columns=lambda x: 'VALUE_' + x, inplace=True)
# # # Read RWK DB for target value (RWK dailly percentage
# #
# rwk_db = pd.read_csv(r"\\ISSHFS.intel.com\ISanalysis$\1274_MAODATA\Config\LITHO\ASML_Scanner\CT\Rework_forML\RW_WET.csv", parse_dates=['OUT_DATE'])
#
# # # rwk_db['OUT_DATE'] = rwk_db.index
# rwk_db['year'] = rwk_db['OUT_DATE'].dt.year
#
# rwk_db['dayofyear'] = rwk_db['OUT_DATE'].dt.dayofyear
#
# rwk_db['key'] = rwk_db['dayofyear'].astype('str') + '_' + rwk_db['year'].astype('str') + '_' + rwk_db['LOT']
#
# rwk_db = rwk_db[['year','dayofyear','LOT','key','REWORK_FLAG']]
#
#
#
#
#
#
# #rwk_db = rwk_db.drop_duplicates(subset=['LOT','year','dayofyear'])
# # rwk_db = rwk_db.reset_index()
#
# #
# # rwk_db['daily_sum'] = rwk_db.groupby(['year','dayofyear','LOT']).apply(lambda x: x.count() if str(x) != ' ' else x)
# #
# # rwk_db['daily_sum_RWK'] = rwk_db.groupby(['year','dayofyear'])['REWORK_FLAG'].transform(lambda x : x.loc[x=='Y'].count())
# #
# # rwk_db['daily_percentage'] = (rwk_db['daily_sum_RWK'] * 100) / rwk_db['daily_sum']
# #
# # rwk_db['rwk_per_lot'] = rwk_db.groupby(['LOT'])['REWORK_FLAG'].transform("count")
# #
# # df = rwk_db.merge(Phys_df, on='LOT').fillna(0)
#
# # apply PCA Trainee
#
# # n_comp = 20
# #
# # print('\nRunning PCA ...')
# # pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
# # X_pca = pca.fit_transform(X)
# # print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())
# #
# # print('Individual variance contributions:')
# # for j in range(n_comp):
# #     print(pca.explained_variance_ratio_[j])
# #
# # pca_samples = pca.transform(X)
# # ps = pd.DataFrame(pca_samples)
# #
# #
# # df_Proj = pd.DataFrame(pca.components_)
# # columns=dict(zip(df_Proj.columns,Phys_df.columns))
# # df_Proj = df_Proj.rename(columns,axis=1)
#
# #
# # conn = sqlite3.connect('my_data.db')
# # ol_df_dcb = pd.read_csv(r"\\ISSHFS.intel.com\ISanalysis$\1274_MAODATA\Litho\ASML\1274_ASML_Dashboard\REG wafer level\V0AtoM0G_1274_Wafer.csv")
# # ol_df_dcb.to_sql('ol_table', conn, if_exists='append' , index = False)
# # rwk_db.to_sql('rwk_tbl', conn, if_exists='append' , index = False)
# # c = conn.cursor()
# # c.execute('''SELECT lot, daily_percentage FROM rwk_tbl''')
# # c.fetchall()
#

#
# # df_all = df_all_gen[df_all_gen['REG_OPERATION'].isin([194182,194187])]
# # test = test.merge(df_all, on= ['LOT'])
# # # test.to_pickle(r"C:\Users\yyramon\Downloads\test.pkl")
# # # ax = pickle.load(open(r"C:\Users\yyramon\Downloads\test.pkl",'rb'))
# #
# #
# #
# # # creating the bar plot , watch for layers distribution
# # gg = df_all_gen[df_all_gen['PERCENT_MEASURED_MIN'] < 30][['REG_OPERATION', 'OVER_LAYER', 'UNDER_LAYER','PERCENT_MEASURED_MIN','OVERLAYER_ENTITY','UNDERLAYER_ENTITY']]
# # gg['OL_UL'] = gg['OVER_LAYER'] + '_' + gg['UNDER_LAYER']
# # gg['SCAN_OL_SCAN_UL'] = gg['OVERLAYER_ENTITY'] + '_' + gg['UNDERLAYER_ENTITY']
# # gg = gg.dropna()
# # # for Layers
# # # Di_layers = gg['OL_UL'].to_list()
# # # gg['per_measured_grouped'] =  gg.groupby('OL_UL')['PERCENT_MEASURED_MIN'].transform('mean')
# # # per_measured = gg['per_measured_grouped'].to_list()
# # # for
# # Di_scanners = gg['SCAN_OL_SCAN_UL'].to_list()
# # gg['per_measured_grouped'] =  gg.groupby('SCAN_OL_SCAN_UL')['PERCENT_MEASURED_MIN'].transform('mean')
# # per_measured = gg['per_measured_grouped'].to_list()
# #
# # fig = plt.figure(figsize=(10, 5))
# # # plt.bar(Di_layers, per_measured, color='maroon',width=0.4)
# # #
# # # plt.xlabel("Di_layers")
# # plt.bar(Di_scanners, per_measured, color='maroon',width=0.4)
# #
# # plt.xlabel("Di_scanners")
# # plt.ylabel("Per_measured")
# # plt.xticks(rotation=90)
# # plt.title("per_measured vs. scanner pairs")
# # plt.show()
# # ^consumes all iterations of the reader: each iteration is a row, composed of a list where each cell value is a list elemnt
# # pickled_data = pickle.dumps(data)
# # restored_data = pickle.loads(pickled_data)
# # csv.writer(open(pathToSaveTo, "wt")).writerows(restored_data)
#
#
# #
# # n_samples = X.shape[0]
# # # We center the data and compute the sample covariance matrix.
# # X -= np.mean(X, axis=0)
# # cov_matrix = np.dot(X.T, X) / n_samples
# # for i,eigenvector in enumerate(pca.components_):
# #     print(i, np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
#
# # split data for f score test
# # eighty_index = int(0.8*len(df.index))
# # df_train = df.loc[df.index <= eighty_index].copy()
# # df_test = df.loc[df.index > eighty_index].copy()
# #
# # X_train = df_train[[x for x in df_train.columns if x.startswith('VALUE') == True]]
# # y_train = df_train['daily_percentage']
# #
# # X_test = df_test[[x for x in df_test.columns if x.startswith('VALUE') == True]]
# # y_test = df_test['daily_percentage']
# #
# #
# # reg = xgb.XGBRegressor(n_estimators=10, max_depth=2)
# # f = reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)
# #
# # plot_importance(reg, height=0.9)
#
#
# # UL OL tool to RWK flag
#
#
