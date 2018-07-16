import mysql.connector
import pandas as pd
from sklearn.preprocessing import scale


def get_data_from_sql_file(file_name):
    with open(file_name, 'r') as sql_file:
        query = sql_file.read()
    cnx = mysql.connector.connect(
        user='f',
        password='Last270731',
        host='127.0.0.1',
        database='bcg'
    )
    sql_result = pd.read_sql(query, cnx)
    # columns = sql_result.columns[1:] # the first column is id not useful for model
    return sql_result




def process_query_result(sql_result, categorical_columns, numeric_columns, process='train'):
    categorical_data = pd.get_dummies(sql_result[categorical_columns])
    categorical_columns = list(categorical_data.columns)
    numeric_data = sql_result[numeric_columns].fillna(0)
    numeric_data_scaled = scale(numeric_data)
    numeric_data_scaled_df = pd.DataFrame(numeric_data_scaled, columns=numeric_columns, dtype=None, copy=False)
    print(type(numeric_data_scaled_df))
    processed_data = pd.concat([categorical_data, numeric_data_scaled_df], axis=1)
    columns = list(categorical_columns + numeric_columns)
    X = processed_data.fillna(.0)[columns[:-1]]
    if process == 'train':
        Y = sql_result['churn'].astype('int')
    else:
        Y = sql_result['id'].apply(lambda x: 0)
    return X, Y


categorical_columns = [
    'activity_new',
    'channel_sales',
    'contract_status',
    'return_cust',
    'is_mod',
    'has_gas',
    'campain_channel',
    'price_seg'
]
numeric_columns = [
    'cons',
    'gas_cons',
    'cons_last_month',
    'days_to_expire',
    'month_since_renewal',
    'contract_length',
    'tenure',
    'forecast_price_energy_p1',
    'forecast_price_energy_p2',
    'forecast_price_pow_p1',
    'forecast_bill_montly',
    'next_month_forecast',
    'forecast_con_monthly',
    'fore_cast_discount',
    'current_paid_cons',
    'invest',
    'num_prod',
    'power_net_margin',
    'total_net_margin',
    'num_years_antig',
    'power_subscribed',
    'p1_power',
    'p2_power',
    'p3_power',
    'p1_energy',
    'p2_energy',
    'p3_energy',
    'p1_power_variance',
    'p2_power_variance',
    'p3_power_variance',
    'p1_energy_variance',
    'p2_energy_variance',
    'p3_energy_variance',
    'price_p1_energy_increate_half_year',
    'price_p2_energy_increate_half_year',
    'price_p3_energy_increate_half_year',
    'price_p1_power_increate_half_year',
    'price_p2_power_increate_half_year',
    'price_p3_power_increate_half_year',
    'price_p1_energy_increate_last_q',
    'price_p2_energy_increate_last_q',
    'price_p3_energy_increate_last_q',
    'price_p1_power_increate_last_q',
    'price_p2_power_increate_last_q',
    'price_p3_power_increate_last_q',
    'price_p1_energy_increate_dec',
    'price_p2_energy_increate_dec',
    'price_p3_energy_increate_dec',
    'price_p1_power_increate_dec',
    'price_p2_power_increate_dec',
    'price_p3_power_increate_dec',
    'churn'
]

if __name__ == '__main__':
    file_name = 'process_train_dataset.sql'
    categorical_columns = ['activity_new', 'channel_sales', 'contract_status', 'is_mod', 'has_gas', 'campain_channel']
    numeric_columns = [
        'cons', 'gas_cons', 'cons_last_month',  'days_to_expire', 'contract_length',
        'forecast_price_energy_p1', 'forecast_price_energy_p2', 'forecast_price_pow_p1', 'forecast_bill_montly',
        'next_month_forecast', 'forecast_con_monthly', 'discount','current_paid_cons', 'num_prod', 'power_net_margin',
        'total_net_margin', 'num_years_antig', 'power_subscribed'
    ]
    sql_result = get_data_from_sql_file(file_name)
    X, Y = process_query_result(sql_result, categorical_columns, numeric_columns)





# cursor = cnx.cursor()
# cursor.execute("SHOW columns FROM train")
# columns = [col[0] for col in cursor.fetchall()]
# print(columns)
# query = 'select * from test limit 100;'
# cursor.execute(query)
# for cur in cursor:
#     print(cur)


folder = 'data/'
train_df = pd.read_csv(folder + 'ml_case_training_data.csv')

train_df['cons'] = train_df['cons_12m'] / 12 # a.cons_12m/12 cons
train_df.drop(['cons_12m'], inplace=True, axis=1)
train_df['gas_cons'] = train_df['cons_gas_12m']  # a.cons_gas_12m/12 gas_cons,
train_df.drop(['cons_gas_12m'], inplace=True, axis=1)
train_df['contract_status'] = train_df['date_end'].apply(lambda x: 'expired' if x < '2016-03-31' else 'not expired')  # case when a.date_end<'2016-03-31' then 'expired' else 'not expired' end contract_status,
# timestampdiff(day, a.date_end, '2016-03-31') days_to_expire,
train_df['week_to_expire'] = (train_df['date_end'].apply(pd.Timestamp) - train_df['date_end'].apply(lambda x: pd.Timestamp('2016-01-01'))).astype('timedelta64[h]')/ (24 * 7)
train_df['week_to_expire'].fillna(999, inplace=True)
train_df['week_to_expire'] = train_df['week_to_expire'].astype('int')
# timestampdiff(month, case when a.date_renewal='' then '2016-03-01' else a.date_renewal end, '2016-03-01') month_since_renewal,
train_df['week_since_renewal'] = (train_df['date_renewal'].apply(lambda x: pd.Timestamp('2016-01-01')) - train_df['date_renewal'].apply(pd.Timestamp)).astype('timedelta64[h]')/ (24 * 7)
train_df['week_since_renewal'].fillna(999, inplace=True)
train_df['week_since_renewal'] = train_df['week_since_renewal'].astype('int')
# timestampdiff(day, a.date_end, '2016-03-31') days_to_expire,
train_df['days_to_expire'] = (train_df['date_end'].apply(pd.Timestamp) - train_df['date_end'].apply(lambda x: pd.Timestamp('2016-01-01')))
# CASE
#   when timestampdiff(month, a.date_modif_prod, a.date_first_activ)<0 then 'yes'
#   when timestampdiff(month, a.date_modif_prod, a.date_activ)<0 then 'yes'
# ELSE 'no' END  is_mod,
# a.id,
# a.activity_new,
# a.channel_sales,
# a.cons_last_month cons_last_month,
# timestampdiff(month, a.date_activ, a.date_end) contract_length,
# case when timestampdiff(month, a.date_activ, a.date_first_activ)<0 then'yes' else 'no' end as return_cust,
# case
# when timestampdiff(month, (a.date_activ), '2016-03-01') > coalesce( timestampdiff(month, (a.date_first_activ), '2016-03-01'), 0)
# then timestampdiff(month, (a.date_activ), '2016-03-01')
# else coalesce( timestampdiff(month, (a.date_first_activ), '2016-03-01'), 0)
# end tenure,
# a.forecast_price_energy_p1,
# a.forecast_price_energy_p2,
# a.forecast_price_pow_p1,
# a.forecast_bill_12m/12 forecast_bill_montly,
# cast((case a.forecast_cons when '' then '0' else a.forecast_cons end)as decimal(10,6)) next_month_forecast,
# a.forecast_cons_12m/12 forecast_con_monthly,
# a.forecast_discount_energy fore_cast_discount,
# a.has_gas,
# a.imp_cons current_paid_cons,
# a.margin_gross_pow_ele ele_gross_margin,
# a.margin_net_pow_ele ele_net_margin,
# a.nb_prod_act num_prod,
# a.net_margin total_net_margin,
# a.num_years_antig,
# a.origin_up campain_channel,
# a.pow_max power_subscribed,
train_df.drop(['date_end'], inplace=True, axis=1)

old_activity_new = [
u'activity_new_amfioduwmscbccofekcfcxpxokiiadpx',
 u'activity_new_eldwcmdwfekwdubwxpuaklxdmkucsdki',
 u'activity_new_uuxeifdawaobxfxxefkdfxkmsmbfoamf',
 u'activity_new_cmdfecexccsmowuoksewsukfcwlplamd',
 u'activity_new_updsxswiffpfixmssiwcfociadowkbsc',
 u'activity_new_cwswmoddxdpulwofkbfdwilbwuodkeme',
 u'activity_new_udmdflpapcfbfpcxbwlbcubxkfoiwaff',
 u'activity_new_fckccskocemwpbxpupsmwalfcibxssbc',
 u'activity_new_mimfwsxwedpuaiciwldiexiakkffbwcb',
 u'activity_new_bdwmomowlswlxwwafowaixacluaiefuf',
 u'activity_new_uaxxxwkppmwfciofupisxsdeauikeppw',
 u'activity_new_aseamwliciokfemccpxaouxwcubucxux',
 u'activity_new_eoakmfoebekdxxupfdbookikcdkkclco',
 u'activity_new_exespdalufcdobebbdlcbmbficidkolw',
 u'activity_new_pkdwiifdckesdafkuoalwpxwkuwpxubb',
 u'activity_new_laxkdmpaielkeuduscppxlwpmaedlaww',
 u'activity_new_ubdxeesfexxaaebslpkoxdbcbmfupuci',
 u'activity_new_xclapbmiasllcspuclsumusikimpckek',
 u'activity_new_fdcwobsbkeeisaofmssomixkmefxslpk',
 u'activity_new_kpkesxdaobicuwwkukxwmdpsbowwbomd',
 u'activity_new_udfcdxexxmdfakwaafplfiplaidudpex',
 u'activity_new_cwouwoubfifoafkxifokoidcuoamebea',
 u'activity_new_ulfmpofsfppwdswdepemulsumksexsbm',
 u'activity_new_dbapibbxclxuuosukfokcodpbplkeclc',
 u'activity_new_xwkiacfesppesmilbxkmbmwdopsmslwp',
 u'activity_new_oolfsafdpblfmubuscwbbuifuxdxkfsd',
 u'activity_new_aiwacxxawfolwlocxdmuwecicpclwpdo',
 u'activity_new_sodadseolpdpoopxaekobldpoixlmeem',
 u'activity_new_dbklukmppmseoekmmxfolmfbdidmawls',
 u'activity_new_kaaekfedeuawdolmeofuailiesofclsb',
 u'activity_new_bddmmiluukefokufkfbsbkwufodoeakc',
 u'activity_new_wcweaxoxmefpfbpfbifcwmfeeubwwkmc',
 u'activity_new_peaesobbauddfswfsudwudcwmeoddoiw',
 u'activity_new_llisdkfcpispsuiwlmpmsxdpwfscdfdx',
 u'activity_new_swbpoldwkuimcalxooacooepdkodcwic',
 u'activity_new_wwbwooasdidfidwldbxdxkamdkaacaxd',
 u'activity_new_fodmmlekiamkemipdukmkkpmdecbbabb',
 u'activity_new_useklixsdbmxwcalbkkbxdfoilisbkcs',
 u'activity_new_kpullxmfmuukcibwibmibwpacpmuabuf',
 u'activity_new_bifsofauudeufolakwmpwwdxxcxeflas',
 u'activity_new_duidibwdmowiudemasiwabceucckesdw',
 u'activity_new_paoauaefwcbedmiowwmokakuisslckbd',
 u'activity_new_uuukommlocffflaemeeppkappcbdpmlw',
 u'activity_new_clafkuosbouwewlexoewscfebeilpfok',
 u'activity_new_fooilksoeuapulaidcawoueldlkbcsca',
 u'activity_new_fxocpcbfplipxiokscwiuexkceoucmko',
 u'activity_new_mcexdumuplwdoisamslcdcmmliwmwbwp',
 u'activity_new_aacewucldmklslcffeckexipaemmsdfk',
 u'activity_new_eamiapdokbfumefocubefudcowecllla',
 u'activity_new_exmccxcauwolkacaceedipbcmodfedfl',
 u'activity_new_mwmuuaeloxbawummwfwcmxckmsfibpwk',
 u'activity_new_cxduoxssmucwpkpsxmiflmicsifosexk',
 u'activity_new_dwdflbsopucwoxdmccmulwiiefiiabel',
 u'activity_new_ckeukxpofbwoacsdeimeoxeuxdblpkwl',
 u'activity_new_waixukdfidxusmdwibmxxkkxbbmbslbf',
 u'activity_new_beosabelsfbxcmawuoicfudpemsxbwxw',
 u'activity_new_dbxlsaldowxpxlxfoueabwbaclmlbuiu',
 u'activity_new_laslwixpcspcffiadlfkeosicpsuaboc',
 u'activity_new_usmcubwfkucbbldluoxfpmaeapwmwiii',
 u'activity_new_oiimbufsmucwemdlmdpwoimsmmalaicf',
 u'activity_new_cxfwwicdxfwpebofockoweifmbxdkkcd',
 u'activity_new_euxklkxpddpscbuoilmisffbxsscmlok',
 u'activity_new_usculwxxobpfompufcclxielfiosluce',
 u'activity_new_xpxablssppduwokmopsaoemoueasdmmm',
 u'activity_new_dsbxbsuowflfmiimikwcdpepuoideeac',
 u'activity_new_awmiwfbewabdaduimwoefiluxdcdwsxb',
 u'activity_new_bbebkcibifdwwepuoclceofdbdipleml',
 u'activity_new_pudpxobkpuxbalsmeimfkwocseoamsoi',
 u'activity_new_wuxsowlmdewcxwfafdwecwsfdbouldex',
 u'activity_new_wmmewikwxokcsaabsuomspccidbawxaf',
 u'activity_new_iullbxlxffasdilkmaxepebuccdmxicc',
 u'activity_new_easulumloioxdxacbacsksmfimsakpxu',
 u'activity_new_klaclcdipfdkebisxwccdbdooobmiwpl',
 u'activity_new_eeaebbffbxdouffsoluusplfcxiadblb',
 u'activity_new_keweuwsssmcfoafcwouubwsobmiocaca',
 u'activity_new_cwkwaxadbfukekuspislmbipbkxdudla',
 u'activity_new_iblsspccddfbbaliawkxfiooeiilfuxc',
 u'activity_new_moaawbkafpwcopipaxsoklsuuoexkaap',
 u'activity_new_xmdofuoxpuaxwuofesefielplxcseubs',
 u'activity_new_feccpaplkaopdipbicfuiacpcieicauo',
 u'activity_new_ukpceuooxfcapeummcoafoixdwexwwdp',
 u'activity_new_aplsmkockmiifibukmmmomommebkdpfk',
 u'activity_new_fcokoocmubsiclsbbefulmfiplksskbf',
 u'activity_new_cfeluxakapclbcismpfoefdmmplddekc',
 u'activity_new_cacmeipkxxdoewfsobspoooxwokpboup',
 u'activity_new_celxmwkmfsefumdlclalblmsecalbpam',
 u'activity_new_fcfeucllkebuploesusiailfleaffdxf',
 u'activity_new_klpmksubowwaicwpcmpiioblpxiwkais',
 u'activity_new_fcwxodkspaloekmowcacfukapocpepxm',
 u'activity_new_mcufpoekpaeboepkkkmoxcmcmlxcwedd',
 u'activity_new_ambaaxsxxwfuspsuabupewfpbbksmcoo',
 u'activity_new_wsboswwcbdfwmufackeodiuooiodoksd',
 u'activity_new_oclxmuppafkockbpkiuksfomiuaeiosm',
 u'activity_new_axspbsdxabbfiplilxkxudsbbiwslocc',
 u'activity_new_bdwasdkxspekskwxikumuebmfufflifx',
 u'activity_new_kmlwkmxoocpieebifumobckeafmidpxf',
 u'activity_new_ewaupfkppoboxiuilledxxlwieawexel',
 u'activity_new_aumipeuxxkfeepiikplpcoifakioeeel',
 u'activity_new_falcdfadiaxaafmplkebedawlaifficp',
 u'activity_new_liekubiseoiflbsuiubbfcwpfsaxbofk',
 u'activity_new_mloddpisebmawpocbcspfiwieodsmded',
 u'activity_new_xlplcsecoeklisfpdmalaxcwekdekdua',
 u'activity_new_ismplispfeepwilxpxmawlekbxiwmxff',
 u'activity_new_ubmsiuoxiaiukxlcfflklluolpeuxaas',
 u'activity_new_fxbibmudumblomsslpiomaxfbiiocbua',
 u'activity_new_upwbxpxmkkbicbamfusaxdloflofisii',
 u'activity_new_iwmoskaicewfewukldfwcdwlxcwwoeom',
 u'activity_new_xscbuwcbpwsilaeadffielubxpfpmpxw',
 u'activity_new_posfbkwcpabfkpkfepfwepwifpubxlpf',
 u'activity_new_xpowpxcwwbecopsbwliawlammdspxipd',
 u'activity_new_kussomukpladumwscxeuuwbeoauibafa',
 u'activity_new_kleicdldcamuaislmkwowllimpblacpf',
 u'activity_new_wkcccadoacxlcoukpxawpopxsfpculmd',
 u'activity_new_lakksuxlbesxeskmemsppkiebumeimkm',
 u'activity_new_bdcmbkmcaussbssoseifimsfkpcuaksk',
 u'activity_new_cficfwpcbaolwaelfwipemswsmobmkfm',
 u'activity_new_mkiiecbapulxwalwiffmsidmikalskif',
 u'activity_new_ieseoxduxoakdspubslseeokdkxiueeo',
 u'activity_new_ompkmxdpkeikxipsowcmceceluupfwde',
 u'activity_new_kimmoxipdxfalcpoueuwkddauubioiwl',
 u'activity_new_olpwwmwiidfiusocbaidbsckxbpoipcu',
 u'activity_new_ixxfakuxuscleimexkbxmeblekcsccbw',
 u'activity_new_wceaopxmdpccxfmcdpopulcaubcxibuw',
 u'activity_new_oxkddxsmiwadlcxkulieewkokxiucsfx',
 u'activity_new_pkakblpskuwxskooaelouomofdulxpdw',
 u'activity_new_fuffsxwkckuoabdsallukmckpwlikakw',
 u'activity_new_eddebmodfooxxwfaslcswiepfmaoxxss',
 u'activity_new_fexixikcmkbfdsexdlmaiswcdxbifsmm',
 u'activity_new_ccpmwcsmadxkpwmofbowesdiepbxioae',
 u'activity_new_lblmkbemolxxlkicccsucmoapesxsplx',
 u'activity_new_ucwixbddwddsslopeflbpmmxpefpxddb',
 u'activity_new_bplwscwaolwimidobccpixabiflaamld',
 u'activity_new_xpwokbdseslumlsislulloalddkioslw',
 u'activity_new_xbwipkcuemuidpumuiomukkicculdmsb',
 u'activity_new_amucfxoxeidufkwdiscepiisxeipuawb',
 u'activity_new_kdmfmpipcmpesudikiemuuldpksxfbff',
 u'activity_new_wwcdlamflfufmxioubuuxpuxkssxkswd',
 u'activity_new_mummxffocmpaikpkxeekmbdxklcikwuc',
 u'activity_new_ckmosmiuspskosddifpslwllwfkscxid',
 u'activity_new_widaskucxiudllboubewwofmciikxswl',
 u'activity_new_umxeuawseikkkxpxsbbafiuwowufsmxs',
 u'activity_new_ikobukxdxwaukmaeoskkkedwmkilpwbk',
 u'activity_new_bilukaxxaslukscimbduakwseilcxupx',
 u'activity_new_lmekuoesfpdmalbikamocsabdlxsdlwm',
 u'activity_new_ommcdoxsluxpfksewskappuxifxxdaaf',
 u'activity_new_iuicsodpwomiidiakdpdkxomecpxcdod',
 u'activity_new_iwueusllaxumufdpwoaewwoipdfsmmsm',
 u'activity_new_xumuokeiidieboawuxkidxufcexecbbl',
 u'activity_new_ooiladiddxkcikildpdfsbwwblcwxxkm',
 u'activity_new_feadeidokwlullcdkcefdafcwkilxauu',
 u'activity_new_iilxdefdkwudppkiekwlcexkdupeucla',
 u'activity_new_ipiiicokaeexiaebwmkecbdummcdmccu',
 u'activity_new_opoiuuwdmxdssidluooopfswlkkkcsxf',
 u'activity_new_beplffiwdfsmiuodulsfscelscscbdix',
 u'activity_new_kcioolmpmuxpoeuicskiafwcmadeflfc',
 u'activity_new_klbickwapkflumbxxksciecxacfbukpi',
 u'activity_new_ddilesxocdoakmfdbmbflfsmeoadpmiw',
 u'activity_new_kllldxcildwkssbmoabmsdffmawsafsf',
 u'activity_new_klsmomiakxdaufoldfilmbxcpuaxiosp',
 u'activity_new_fswbowspcfidbaiuuwuicmmxdffupofp',
 u'activity_new_ksukukiwxdxwbfwaapmuwippflemumlp',
 u'activity_new_wkwdccuiboaeaalcaawlwmldiwmpewma'
]

old_campaign_channel = [
    u'campain_channel_usapbepcfoloekilkwsdiboslwaxobdp',
    u'campain_channel_ewxeelcelemmiwuafmddpobolfuxioce',
]
new_activity_new = [u'activity_new_mbdpueaepmxiidaadsixoemwxwxexkwd',
 u'activity_new_iilkwmilepwlosubloxwelwwbxeoodac',
 u'activity_new_fpacspcuiolofiuceulclpulmoxccixf',
 u'activity_new_umkamlxffsefucodlxelwpklcbfxdokx',
 u'activity_new_ecmcesabbpeuxoamxbmxxibalosaiool',
 u'activity_new_sxdoeakpwoxmmlalilcwpxkmicfcxdbb',
 u'activity_new_bxbkdblfpallcawfcfuecaxlaalimica',
 u'activity_new_uxiuocumlpopmlisxbfmepusckfodado',
 u'activity_new_lsiesslxfuexwslpccllmelfdbomopdo',
 u'activity_new_mpeaasloolumallpddxxemfclwwpaaku',
 u'activity_new_ppxfxpxokmmdldubflwiuiekksdmeowc',
 u'activity_new_oxpsiwbbacuuubpmbxbskcaexpicwusu',
 u'activity_new_bibcwkdkxsbadsecsuwudubbsfeexemw',
 u'activity_new_kfpdkiweeadupwbbdbacbsoakidcwmpi',
 u'activity_new_fkisexucladmbidloukpbbiaaxlokdxp',
 u'activity_new_ikofkbbsefbpasiomapclxkddisloube',
 u'activity_new_kwsddfuscoppaolpfdpukcpdkbiauaaw',
 u'activity_new_lfdmkwouxeflfsaomoobfwcsifudcwam',
 u'activity_new_amdoluxlsxfbsamsxufdwmdmcalebmmm',
 u'activity_new_komeelowixlbkxdkwelbdcfopldioesk',
 u'activity_new_maecxiwoobpkdsibceklmbolfkbabebi',
 u'activity_new_ceemdsblcexoxxwfdlkpxweplbbfwwao',
 u'activity_new_bmlxiluuofldcelibslwulsbomkoxldo',
 u'activity_new_sbmbmepbdsaduwbumxcibifiblauwcbo',
 u'activity_new_dwbkawecemscwofelcpwdseaauaefxmm',
]

new_campaign_channel = [
u'campain_channel_aabpopmuoobccoxasfsksebxoxffdcxs',
'invest',
'price_p3_power_increate_dec'
]


