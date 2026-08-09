# Copyright (c) 2026 Kohei Shintani. Licensed under CC BY-NC 4.0
# (Attribution-NonCommercial): https://creativecommons.org/licenses/by-nc/4.0/
# Commercial use requires prior permission (see LICENSE in the source repository).
# This file has been minified for distribution (comments/docstrings removed).
_h='class_value'
_g='datetime'
_f='default'
_e='target'
_d='onehot'
_c='composite_target'
_b='models'
_a='blend_meta.pkl'
_Z='lgbm_bag'
_Y='lgbm_bag_meta.json'
_X='method'
_W='mlp_model.pkl'
_V='gp_model.pkl'
_U='model'
_T='scaler'
_S='linear_model.pkl'
_R='xt'
_Q='rf'
_P='mlp'
_O='gp'
_N='linear'
_M='lgbm'
_L='coerce'
_K='feature_name'
_J='source_col'
_I=False
_H='rb'
_G='medians'
_F='__NaN__'
_E='utf-8'
_D=1.
_C=None
_B=True
_A='feat_cols'
import sys,os,re,json,math,pathlib,pickle,numpy as np,pandas as pd
_DATETIME_RE=re.compile('^(\\d{4})([-/])(\\d{2})\\2(\\d{2})(?:[ T]?(\\d{2}):(\\d{2})(?::(\\d{2}))?)?$')
_DATETIME_DAYS_IN_MONTH=31,29,31,30,31,30,31,31,30,31,30,31
_DATETIME_PART_INDEX={'hour':0,'dow':1,'month':2,'epoch_days':3}
def _days_from_civil(y,m,d):'Howard Hinnant の days_from_civil。train_bridge.py と同一実装(参照日テストで\n    C++/JS/train_bridge.pyと全一致確認済み、scratchpad dt_unittest.*参照)。';y=y-(1 if m<=2 else 0);B=(y if y>=0 else y-399)//400;A=y-B*400;C=(153*(m+(-3 if m>2 else 9))+2)//5+d-1;D=A*365+A//4-A//100+C;return B*146097+D-719468
def _weekday_from_days(days):return(days+3)%7
def _parse_datetime_parts(s):
	'train_bridge._parse_datetime_parts と同一実装。成功なら\n    (hour, dow, month, epoch_days) の4-tuple、失敗ならNoneを返す。';A=_DATETIME_RE.match(s.strip())
	if not A:return
	F,B,C=int(A.group(1)),int(A.group(3)),int(A.group(4));D=int(A.group(5))if A.group(5)else 0;G=int(A.group(6))if A.group(6)else 0
	if not 1<=B<=12 or not 1<=C<=31 or not 0<=D<=23 or not 0<=G<=59:return
	if C>_DATETIME_DAYS_IN_MONTH[B-1]:return
	E=_days_from_civil(F,B,C);H=_weekday_from_days(E);return D,H,B,E
NUMKEY_SEP='\x1f'
def _canon_numeric_key_part(v):
	'実バグ2026-08(M4): 丸め規則がPython(banker\'s round)/JS(Math.round=half-up)/\n    C++(half-away-from-zero)で三者三様だった(例: 2.5→Py"2"/JS"3"/C++"3"、\n    -2.5→Py"-2"/JS"-2"/C++"-3")ため、下方で定義済みの_round_half_away\n    (half-away-from-zero、round_output後処理と共通、スカラーにも使える)で統一する。'
	if v is _C:return _F
	try:A=float(v)
	except(TypeError,ValueError):return _F
	if not np.isfinite(A)or abs(A)>=1e15:return _F
	return str(int(_round_half_away(A)))
def _build_composite_key(source_cols,row):return NUMKEY_SEP.join(_canon_numeric_key_part(row.get(A))for A in source_cols)
def _read_csv_with_encoding_fallback(csv_path,dtype=_C):
	'まずUTF-8として読み込みを試み、デコードできなければ Shift-JIS(cp932) として読む。\n    日本語Excelが既定で書き出すShift-JIS CSVがUTF-8として「�」化けしたまま\n    サイレントに予測が完走してしまうのを防ぐ(中-7)。\n    低-M21: 以前はUTF-8妥当性の事前検査のためファイル全体を素のバイト列として読み込み\n    (`f.read()`でメモリに全展開)、その上でさらにpandasにも読ませていたため、大きな\n    CSVで実質2回分のI/O・デコードが走っていた。pd.read_csv自体もutf-8で全文デコード\n    するので不正バイトがあれば同じくUnicodeDecodeErrorを送出する。それをそのまま\n    フォールバック判定に使えば1回の読み込みで済む。\n    dtype: 実バグ2026-08(M2)対策。onehot/target encodingのソース列は生セル文字列で\n    キー照合するため、呼び出し側が該当列を dtype=str で強制指定できるようにする\n    (未指定なら従来通りpandas自動推定)。';C=dtype;B=csv_path
	try:A=pd.read_csv(B,encoding=_E,dtype=C)
	except UnicodeDecodeError:print(f"[Robot] CSVがUTF-8として不正 → Shift-JIS(cp932)として読み込みます",flush=_B);A=pd.read_csv(B,encoding='cp932',dtype=C)
	A.columns=A.columns.str.strip();A=A.replace([np.inf,-np.inf],np.nan);return A
def _sanitize_json(obj):
	"dict/list を再帰し、非有限 float を None に置換する（train_bridge.py と同一実装）。\n    列レベルの apply では NoneがSeries再構成時にNaNへ戻ってしまう（高-M2）ため、\n    to_dict('records') 後の生の dict/list に対して再帰的に適用する。";A=obj
	if isinstance(A,dict):return{A:_sanitize_json(B)for(A,B)in A.items()}
	if isinstance(A,(list,tuple)):return[_sanitize_json(A)for A in A]
	if isinstance(A,(float,np.floating)):return float(A)if math.isfinite(float(A))else _C
	if isinstance(A,(np.integer,)):return int(A)
	return A
def _invert_y(arr,transform,params):
	F=transform;E=arr
	if F=='log1p':return np.expm1(E)
	if F=='yeo_johnson':
		B=np.asarray(E,float);C=float(params.get('lambda',_D));D=np.empty_like(B);A=B>=0
		if abs(C)<1e-06:D[A]=np.expm1(B[A])
		else:G=np.clip(C*B[A]+_D,1e-12,_C);D[A]=G**(_D/C)-_D
		if abs(C-2.)<1e-06:D[~A]=_D-np.exp(-B[~A])
		else:H=np.clip(-(2.-C)*B[~A]+_D,1e-12,_C);D[~A]=_D-H**(_D/(2.-C))
		return D
	return E
def _round_half_away(arr):'half-away-from-zero 丸め。native exe の std::round と一致させる。';A=arr;A=np.asarray(A,dtype=float);return np.copysign(np.floor(np.abs(A)+.5),A)
def _fill_values(fc,model_medians,impute_medians):'モデル自身の学習時 median を優先し、なければ全列 median にフォールバック。';return{A:model_medians.get(A,impute_medians.get(A,.0))for A in fc}
def _build_feature_matrix(df,fc,med,impute_medians):'指定列を数値行列として取り出す。数値として不正な文字列(例: "12abc")が\n    混入したセルは pd.to_numeric(errors="coerce") で NaN 化してから median 補完する\n    (欠損セルと同じ経路で扱う)。以前は reindex 後にいきなり .values.astype(float) を\n    呼んでおり、そのようなセルが1つでもある列があると ValueError で予測全体が\n    クラッシュしていた。native exe は std::stod の部分パース(例: "12abc"→12として\n    使ってしまう)、Web版JSは Number() ベースで NaN 化(=本関数と同じ挙動)と、\n    実装ごとに挙動が食い違っていた(中-10)。native側もこの関数と同じ「非数値→NaN→\n    median補完」に合わせて修正済み(native_predictor/predict_native_v2.cpp)。';A=df.reindex(columns=fc).apply(pd.to_numeric,errors=_L);return A.fillna(_fill_values(fc,med,impute_medians)).values.astype(float)
def _load_lgbm_meta(model_dir):
	A=os.path.join(model_dir,'lgbm_meta.json')
	if os.path.exists(A):
		with open(A,encoding=_E)as B:return json.load(B)
def _predict_lgbm(df,model_dir,feat_cols,impute_medians,y_transform,y_params):
	C=feat_cols;B=model_dir;import lightgbm as F;G=F.Booster(model_file=os.path.join(B,'lgbm_model.txt'));A=_load_lgbm_meta(B)
	if A:D=A.get(_A,C);E=A.get(_G,{})
	else:D,E=C,{}
	H=_build_feature_matrix(df,D,E,impute_medians);return _invert_y(G.predict(H),y_transform,y_params)
def _predict_linear(df,model_dir,impute_medians,y_transform,y_params):
	D=y_params;C=y_transform
	with open(os.path.join(model_dir,_S),_H)as F:A=pickle.load(F)
	G=A[_A];H=A.get(_G,{});E=_build_feature_matrix(df,G,H,impute_medians)
	if A.get('use_poly'):B=A[_T].transform(E);return _invert_y(A[_U].predict(A['poly'].transform(B)),C,D)
	B=A[_T].transform(E);return _invert_y(A[_U].predict(B),C,D)
def _predict_gp(df,model_dir,impute_medians,y_transform,y_params):
	with open(os.path.join(model_dir,_V),_H)as B:A=pickle.load(B)
	C=A[_A];D=A.get(_G,{});E=_build_feature_matrix(df,C,D,impute_medians);F=A[_T].transform(E);return _invert_y(A[_U].predict(F),y_transform,y_params)
def _predict_mlp(df,model_dir,impute_medians,y_transform,y_params):
	with open(os.path.join(model_dir,_W),_H)as B:A=pickle.load(B)
	C=A[_A];D=A.get(_G,{});E=_build_feature_matrix(df,C,D,impute_medians);return _invert_y(A['pipeline'].predict(E),y_transform,y_params)
def _predict_lgbm_bag(kind,df,model_dir,impute_medians,y_transform,y_params):
	'LightGBM バギング多様化メンバー（RF/XT モード。テキスト形式 + sidecar meta）。';A=model_dir;import lightgbm as F;G=F.Booster(model_file=os.path.join(A,f"{kind}_model.txt"));B=os.path.join(A,f"{kind}_meta.json")
	if os.path.exists(B):
		with open(B,encoding=_E)as H:C=json.load(H)
		D=C.get(_A,[]);E=C.get(_G,{})
	else:D,E=[],{}
	I=_build_feature_matrix(df,D,E,impute_medians);return _invert_y(G.predict(I),y_transform,y_params)
def _predict_lgbm_foldbag(df,model_dir,impute_medians,y_transform,y_params):
	'対策(2026-07 第2弾・真因①): LightGBM fold バギング(train_bridge._try_lgbm_steps が\n    書き出す lgbm_model_fold{k}.txt の等重み平均)。native/JS は .treg 上で通常の blend\n    (K個のlgbm型メンバー、重み1/K)として読むため変更不要だが、predict_template.py は\n    .treg を読まず trained_model/ の実体ファイルを直接読む独立実装のため専用関数が要る。';B=model_dir;import lightgbm as E
	with open(os.path.join(B,_Y),encoding=_E)as F:A=json.load(F)
	G,H,C=A[_A],A.get(_G,{}),A['n_folds'];I=_build_feature_matrix(df,G,H,impute_medians);D=np.zeros(len(df))
	for J in range(C):K=E.Booster(model_file=os.path.join(B,f"lgbm_model_fold{J}.txt"));D+=_invert_y(K.predict(I),y_transform,y_params)
	return D/C
def _predict_by_type(model_type,df,model_dir,feat_cols_from_meta,impute_medians,y_transform,y_params):
	F=y_params;E=y_transform;D=impute_medians;C=model_dir;B=df;A=model_type
	if A==_M:return _predict_lgbm(B,C,feat_cols_from_meta,D,E,F)
	if A==_N:return _predict_linear(B,C,D,E,F)
	if A==_O:return _predict_gp(B,C,D,E,F)
	if A==_P:return _predict_mlp(B,C,D,E,F)
	if A in(_Q,_R):return _predict_lgbm_bag(A,B,C,D,E,F)
	if A==_Z:return _predict_lgbm_foldbag(B,C,D,E,F)
	raise ValueError(f"未知のモデル種別: {A}")
_NAME_TO_TYPE={'Linear (Ridge)':_N,'LightGBM':_M,'GaussianProcess (ARD-RBF)':_O,'MLP':_P,'LGBM-RF':_Q,'LGBM-XT':_R}
def _apply_derived(df,recipe):
	'学習時の派生特徴レシピを適用する（train_bridge._apply_derived と同一仕様）。\n    ソース欠損・非有限は NaN として伝播し、各モデルの median 補完に委ねる。';G=recipe;F='op';A=df
	if not G:return A
	A=A.copy();H=pd.Series(np.nan,index=A.index)
	for C in G:
		B=C.get('cols',[]);D=pd.to_numeric(A[B[0]],errors=_L)if B and B[0]in A.columns else H
		if C.get(F)=='mul':I=pd.to_numeric(A[B[1]],errors=_L)if len(B)>1 and B[1]in A.columns else H;E=(D*I).values.astype(float)
		elif C.get(F)=='sq':E=(D*D).values.astype(float)
		elif C.get(F)=='sign':E=np.sign(D.values.astype(float))
		else:continue
		A[C['name']]=np.where(np.isfinite(E),E,np.nan)
	return A
def _predict_blend(df,model_dir,feat_cols_from_meta,impute_medians,y_transform,y_params):
	F=model_dir
	with open(os.path.join(F,_a),_H)as J:A=pickle.load(J)
	K=A[_b];L=A.get('version',1)>=2 or A.get('normalize',_B)is _I;G=[];B=[]
	for C in K:
		H=_NAME_TO_TYPE.get(C)
		if H is _C:raise RuntimeError(f"Blend の未知サブモデル '{C}'")
		M=_predict_by_type(H,df,F,feat_cols_from_meta,impute_medians,y_transform,y_params);G.append(M);B.append(C);print(f"[Robot] Blend サブモデル '{C}' 完了",flush=_B)
	I=np.column_stack(G);N=A.get('weights',{});D=np.array([N.get(A,.0)for A in B],dtype=float)
	if L:return I@D
	E=D.sum()
	if E<=0:D=np.ones(len(B));E=float(len(B))
	return(I*D).sum(axis=1)/E
def _collect_required_cols(model_type,model_dir,meta):
	'予測に必要な入力列（モデル実使用列）を集める。blend は全サブモデルの union。';E=model_dir;C=meta;B=model_type
	def F(fname):
		A=os.path.join(E,fname)
		if not os.path.exists(A):return[]
		try:
			with open(A,_H)as B:return pickle.load(B).get(_A,[])
		except Exception:return[]
	def G(fname):
		A=os.path.join(E,fname)
		if not os.path.exists(A):return[]
		try:
			with open(A,encoding=_E)as B:return json.load(B).get(_A,[])
		except Exception:return[]
	if B=='blend':
		A=[]
		try:
			with open(os.path.join(E,_a),_H)as L:I=pickle.load(L).get(_b,[])
		except Exception:I=[]
		for M in I:
			D=_NAME_TO_TYPE.get(M)
			if D==_M:H=_load_lgbm_meta(E);A+=(H or{}).get(_A,C.get(_A,[]))
			elif D==_N:A+=F(_S)
			elif D==_O:A+=F(_V)
			elif D==_P:A+=F(_W)
			elif D in(_Q,_R):A+=G(f"{D}_meta.json")
		J=set();return[A for A in A if not(A in J or J.add(A))]
	if B==_M:H=_load_lgbm_meta(E);return(H or{}).get(_A,C.get(_A,[]))
	if B in(_Q,_R):A=G(f"{B}_meta.json");return A if A else C.get(_A,[])
	if B==_Z:A=G(_Y);return A if A else C.get(_A,[])
	K={_N:_S,_O:_V,_P:_W}.get(B)
	if K:A=F(K);return A if A else C.get(_A,[])
	return C.get(_A,[])
def _to_raw_required(required_cols,recipe,cat_encoders=_C):
	'必要列のうち派生特徴をそのソース列（CSV に実在すべき列）へ展開する。\n    精度レバー4: カテゴリエンコーダの生成列(one-hot indicator名やtarget-encoding後の\n    元列名)もさらに source_col まで1段解決する(native/JS版 raw_sources_for と同一仕様)。\n    v7/真因②対策: composite_targetはsource_colがNUMKEY_SEPで連結した複数のCSV列名の\n    ため、1列名ではなく複数列名のリストに展開する。';C={A['name']:A for A in recipe};D={}
	for A in cat_encoders or[]:E=A[_J];D[A[_K]]=E.split(NUMKEY_SEP)if A.get(_X)==_c else[E]
	F,G=[],set()
	for A in required_cols:
		I=C[A].get('cols',[])if A in C else[A]
		for H in I:
			for B in D.get(H,[H]):
				if B and B not in G:G.add(B);F.append(B)
	return F
if __name__=='__main__':
	sys.stdout.reconfigure(line_buffering=_B,encoding=_E)
	if len(sys.argv)<2:sys.exit(1)
	csv_path=sys.argv[1];print(f"[Robot] CSV 読み込み: {os.path.basename(csv_path)}",flush=_B);model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'trained_model');meta_path=os.path.join(model_dir,'model_meta.json')
	with open(meta_path,encoding=_E)as f:meta=json.load(f)
	model_type=meta['model_type'];feat_cols=meta[_A];target_col=meta['target_col'];y_transform=meta.get('y_transform','none');y_params=meta.get('y_params',{});cat_encoders=meta.get('cat_encoders',{});x_clip=meta.get('x_clip',{});postprocess=meta.get('postprocess',{});smear=postprocess.get('smear',_D);y_clip=postprocess.get('y_clip',[-3.4e38,3.4e38]);_str_dtype_cols={A[_J]for A in cat_encoders or[]if A.get(_X)in(_d,_e)and A.get(_J)};_dtype_override={A:str for A in _str_dtype_cols}if _str_dtype_cols else _C;df=_read_csv_with_encoding_fallback(csv_path,dtype=_dtype_override);df_model=df.copy();round_output=postprocess.get('round_output',_I);print(f"[Robot] モデル: {meta.get('model_label',model_type)}",flush=_B)
	if cat_encoders:
		applied_cols=set()
		for spec in cat_encoders:
			method=spec.get(_X)
			if method==_c:
				source_cols=spec[_J].split(NUMKEY_SEP)
				if not all(A in df_model.columns for A in source_cols):continue
				applied_cols.update(source_cols);m,default=spec.get('map',{}),spec.get(_f,.0);df_model[spec[_K]]=df_model.apply(lambda row,m=m,d=default,sc=source_cols:m.get(_build_composite_key(sc,row),d),axis=1);continue
			col=spec.get(_J);col_missing=col not in df_model.columns
			if col_missing and method==_g:continue
			applied_cols.add(col)
			if method==_d:
				if col_missing:df_model[spec[_K]]=float(spec[_h]==_F)
				else:s_filled=df_model[col].fillna(_F).astype(str);df_model[spec[_K]]=(s_filled==spec[_h]).astype(float)
			elif method==_g:
				part_idx=_DATETIME_PART_INDEX[spec['part']]
				def _extract_part(v,part_idx=part_idx):
					if pd.isna(v):return np.nan
					A=_parse_datetime_parts(str(v));return np.nan if A is _C else float(A[part_idx])
				df_model[spec[_K]]=df_model[col].map(_extract_part)
			else:
				m,default=spec.get('map',{}),spec.get(_f,.0)
				if col_missing:df_model[col]=float(m.get(_F,default))
				else:s_filled=df_model[col].fillna(_F).astype(str);df_model[col]=s_filled.map(lambda v,m=m,d=default:m.get(v,d)).astype(float)
		if applied_cols:print(f"[Robot] カテゴリ列エンコード: {sorted(applied_cols)}",flush=_B)
	if x_clip:
		for(col,bounds)in x_clip.items():
			if col in df_model.columns:lo,hi=bounds[0],bounds[1];df_model[col]=pd.to_numeric(df_model[col],errors=_L).clip(lower=lo,upper=hi)
	derived_recipe=meta.get('derived_features',[])or[]
	if derived_recipe:df_model=_apply_derived(df_model,derived_recipe);print(f"[Robot] 自動特徴量 {len(derived_recipe)} 本を再計算",flush=_B)
	impute_medians={};impute_path=os.path.join(model_dir,'impute_medians.json')
	if os.path.exists(impute_path):
		with open(impute_path,encoding=_E)as f:impute_medians=json.load(f)
	required_cols=_to_raw_required(_collect_required_cols(model_type,model_dir,meta),derived_recipe,cat_encoders);missing_cols=[A for A in required_cols if A not in df.columns]
	if missing_cols:print(f"[Robot] 警告: 学習時の列がCSVにありません → median補完で続行: {missing_cols}",flush=_B)
	try:
		if model_type=='blend':preds_arr=_predict_blend(df_model,model_dir,feat_cols,impute_medians,y_transform,y_params)
		else:preds_arr=_predict_by_type(model_type,df_model,model_dir,feat_cols,impute_medians,y_transform,y_params)
	except Exception as e:print(f"[Robot] PREDICT_ERROR:predict_failed:{json.dumps({'detail':str(e)},ensure_ascii=_I)}",flush=_B);sys.exit(1)
	preds_arr=np.asarray(preds_arr,dtype=float)*smear;preds_arr=np.clip(preds_arr,y_clip[0],y_clip[1])
	if round_output:preds_arr=_round_half_away(preds_arr)
	preds=pd.Series(preds_arr,index=df.index);df[target_col]=preds;in_path=pathlib.Path(csv_path);out_path=in_path.parent/(in_path.stem+'_predicted'+in_path.suffix);df.to_csv(str(out_path),index=_I,encoding='utf-8-sig');print(f"[Robot] 保存完了: {out_path.name}",flush=_B);preview=df.head(500).to_dict('records');result={'rows':len(df),'mean':round(float(preds.mean()),2)if len(preds)>0 else .0,'std':round(float(preds.std()),2)if len(preds)>1 else .0,_e:target_col,'columns':list(df.columns),'preview':preview,'output_path':str(out_path),'output_name':out_path.name,'missing_cols':missing_cols};result=_sanitize_json(result);print(f"PREDICT_JSON:{json.dumps(result,ensure_ascii=_I,default=str,allow_nan=_I)}",flush=_B);print('[Robot] 完了。',flush=_B)