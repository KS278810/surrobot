# Copyright (c) 2026 Kohei Shintani. Licensed under CC BY-NC 4.0
# (Attribution-NonCommercial): https://creativecommons.org/licenses/by-nc/4.0/
# Commercial use requires prior permission (see LICENSE in the source repository).
# This file has been minified for distribution (comments/docstrings removed).
_AF='class_value'
_AE='lgbm_bag_meta.json'
_AD='blend_meta.pkl'
_AC='model.treg'
_AB='payload'
_AA='export_type'
_A9='members'
_A8='composite_target'
_A7='linear_poly'
_A6='pipeline'
_A5='bag_n_folds'
_A4='num_leaves'
_A3='target'
_A2='default'
_A1='datetime'
_A0='lgbm_meta.json'
_z='regression'
_y='col'
_x='weight'
_w='model_txt'
_v='yeo-johnson'
_u='mlp_model.pkl'
_t='gp_model.pkl'
_s='lgbm_model.txt'
_r='n_estimators'
_q='use_poly'
_p='lambda'
_o='rb'
_n='xt'
_m='val'
_l='yeo_johnson'
_k='log1p'
_j='linear_model.pkl'
_i='cols'
_h='op'
_g=' '
_f='method'
_e='rf'
_d='blend'
_c='target_col'
_b='gp'
_a='linear'
_Z='model'
_Y='mlp'
_X='lgbm'
_W='coerce'
_V='oof'
_U='scaler'
_T='wb'
_S='feature_name'
_R='utf-8'
_Q='exportable'
_P='source_col'
_O='<I'
_N='used_cols'
_M='none'
_L='train'
_K='eval_kind'
_J='train_r2'
_I='feat_cols'
_H='pct'
_G='name'
_F='medians'
_E=1.
_D=.0
_C=False
_B=True
_A=None
import sys,os,re
try:_NUM_JOBS=int(sys.argv[5])if len(sys.argv)>5 else 4
except(ValueError,IndexError):_NUM_JOBS=4
_NUM_JOBS=max(1,min(_NUM_JOBS,32))
for _env_key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_env_key]=str(_NUM_JOBS)
import json,shutil,math,struct,asyncio,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
from _light import StratifiedKFold,KFold
from _light import r2_score,mean_squared_error,mean_absolute_error
try:
	import threadpoolctl as _threadpoolctl
	def _thread_limit(n):return _threadpoolctl.threadpool_limits(limits=max(1,int(n)))
except Exception:
	import contextlib
	def _thread_limit(n):return contextlib.nullcontext()
MIN_ROWS_FOR_SPLIT=10
INSTANT_MAX_TRAIN_ROWS=500
INSTANT_SAMPLE_SEED=20260730
GP_MAX_TRAIN=300
SCIPY_GP_MIN_ROWS=10
MLP_MIN_ROWS=30
OUTLIER_IQR_MULT=3.
OUTLIER_IQR_QUICK=4.5
SKEW_THRESH=.75
X_CLIP_PCTILE=_E,99.
MAX_MISS_RATE=.7
POLY_MAX_ROWS=200
POLY_MAX_FEATS=8
BLEND_R2_THRESH=.3
CONST_STD_EPS=1e-12
DUP_CORR_THRESH=.999
SMALL_N_OOF_THRESH=50
Y_CLIP_MARGIN_FRAC=.05
SMEAR_CLIP_RANGE=.5,2.
LGBM_HALVING_FOLDS=2
LGBM_SCREEN_MIN_FEATS=10
X_CLIP_SENTINEL=3.4e38
ES_VAL_FRAC=.1
ES_SPLIT_SEED=20260716
ES_MIN_TRAIN_ROWS=20
LGBM_FINALISTS=3
MLP_N_CANDIDATES=8
MLP_FINALISTS=2
MLP_HALVING_FOLDS=2
LGBM_PARAM_QUICK=dict(num_leaves=31,learning_rate=.05,n_estimators=500)
MLP_PARAM_QUICK=dict(alpha=.0001,single_layer=_B)
BLEND_MARGIN=.005
STACKER_MARGIN=.01
FE_MIN_ROWS=50
FE_MAX_RAW=12
FE_TOP_K=15
def _lgbm_search_candidates(n_rows,rng_seed=42):
	'LightGBM のランダムサーチ候補。先頭は現行デフォルト（安全な基準線）。\n    候補数はデータ量に応じて自動調整する。';D=n_rows;B=24
	if D>5000:B=14
	if D>20000:B=8
	A=np.random.RandomState(rng_seed);C=[dict(num_leaves=31,learning_rate=.05,n_estimators=3000)]
	while len(C)<B:C.append(dict(num_leaves=int(A.choice([7,15,31,63,127])),learning_rate=float(A.choice([.01,.02,.03,.05,.1])),n_estimators=3000,min_child_samples=int(A.choice([5,10,20,40])),colsample_bytree=float(np.round(A.uniform(.5,_E),2)),subsample=float(np.round(A.uniform(.6,_E),2)),reg_alpha=float(np.round(10**A.uniform(-3,.5),4)),reg_lambda=float(np.round(10**A.uniform(-2,1.5),4)),max_bin=int(A.choice([63,127,255,511])),extra_trees=bool(A.choice([_C,_B]))))
	return C
def _mlp_search_candidates(rng_seed=42):
	'MLP のランダムサーチ候補。先頭3つは現行グリッド（後方互換の基準線）。';E='triple';D='single';A=np.random.RandomState(rng_seed);B=[dict(alpha=.0001),dict(alpha=.01,single_layer=_B),dict(alpha=1e-05,extra_layer=_B)]
	while len(B)<MLP_N_CANDIDATES:C=A.choice([D,'double',E]);B.append(dict(alpha=float(10**A.uniform(-6,-1)),single_layer=C==D,extra_layer=C==E,width=float(A.choice([.5,_E,2.]))))
	return B
GP_RESTARTS_THOROUGH=3
GP_RESTARTS_QUICK=1
def _emit_progress(pct,key,*B):'UI 進捗バー用のマイルストーン通知。lib.rs が log_data として素通しし、\n    フロントが `PROGRESS:` prefix を判定してバー幅とサブラベルを更新する。\n    labelは表示用の日本語文ではなくキー(+可変パラメータ)で送る。フロント側が\n    辞書引きして表示言語に翻訳する。パラメータは現状すべて非負整数(候補No/候補総数)\n    のみ。コロンを含み得る自由文字列(列名や例外メッセージ)は絶対に渡さないこと\n    (ERRORキー化と違いJSONエスケープしていないため)。';A=pct;A=int(max(0,min(100,A)));C=':'.join([key,*map(str,B)]);print(f"PROGRESS:{A}:{C}",flush=_B)
def _error_exit(key,**A):'ERROR: + キー + JSONパラメータで終了する。`ERROR:`prefix自体は不変のため\n    lib.rs/surrobot-engine.js/offline-engine.jsのprefix判定は無改修で機能する。\n    列名や例外メッセージなど任意の文字(コロン含む)が混在するパラメータは\n    json.dumpsでエスケープする(改行は\\nの2文字に変換されるため行ベースの\n    プロトコルは壊れない)。';print(f"ERROR:{key}:{json.dumps(A,ensure_ascii=_C)}",flush=_B);sys.exit(1)
_IS_PYODIDE=sys.platform=='emscripten'
async def _maybe_yield():
	'Pyodide実行時のみ、ブラウザに一度制御を返す(ロボGIF/進捗描画のための息継ぎ)。\n    Web版はPyodideの学習処理がメインスレッド上で同期的にCPUを占有し続けるため(offline.html。\n    HTTP配信版はWeb Worker化済みで本関数の効果は無いが害もない)、await asyncio.sleep(0)で\n    定期的にJSのイベントループへ制御を返し、候補/fold単位で描画・応答の機会を作る。\n    exe版はCPython(_IS_PYODIDE=False)なので何もしない。'
	if _IS_PYODIDE:await asyncio.sleep(0)
def _round_half_away(arr):'half-away-from-zero 丸め。C++ 側 std::round と一致させる (np.round は銀行家丸め)。';A=arr;A=np.asarray(A,dtype=float);return np.copysign(np.floor(np.abs(A)+.5),A)
def _sanitize_json(obj):
	'dict/list を再帰し、非有限 float を None に置換する (serde_json は NaN/Inf を拒否する)。';A=obj
	if isinstance(A,dict):return{A:_sanitize_json(B)for(A,B)in A.items()}
	if isinstance(A,(list,tuple)):return[_sanitize_json(A)for A in A]
	if isinstance(A,(float,np.floating)):return float(A)if math.isfinite(float(A))else _A
	if isinstance(A,(np.integer,)):return int(A)
	return A
def _y_true_for(eval_kind,y_raw_all,tr_idx0,va_idx0):
	'winsorize(外れ値クリップ)前の生yから、評価対象行を取り出す。\n    以前は winsorize 後の df/df_train/df_val から取っており、学習側で外れ値を\n    丸め込んだ後の「甘くなった正解値」に対してR²/RMSE/MAEを計算していたため、\n    実データに対する精度より楽観的な数値が出ていた(性能アップ計画Phase2/評価の\n    楽観バイアス低減)。y_raw_all は df と同じ行順序を保つ(winsorizeは値の書き換えのみで\n    行の並べ替え・削除は行わないため、tr_idx0/va_idx0とそのまま対応する)。';C=va_idx0;B=y_raw_all;A=eval_kind
	if A==_V:return B
	if A==_m and C is not _A and len(C)>0:return B[C]
	if A==_L:return B[tr_idx0]
def _eval_metrics(val_preds,eval_kind,y_raw_all,tr_idx0,va_idx0):
	'評価指標（RMSE/MAE/eval_on/eval_rows/y_true）を計算する。eval_kind は明示指定。\n    中-M4: 評価不能時は「誤差ゼロ」に見える0.0ではなくNoneを返す(_sanitize_json経由で\n    JSON上はnullとして出力され、フロントで「評価できませんでした」等の扱いが可能になる)。';D=tr_idx0;C=eval_kind;B=val_preds;A=_y_true_for(C,y_raw_all,D,va_idx0)
	if B is _A or A is _A or len(A)!=len(B):return _A,_A,C or _L,len(D),_A
	E=float(np.sqrt(mean_squared_error(A,B)));F=float(mean_absolute_error(A,B));return E,F,C,len(A),A
def _candidate_r2_std(preds_arr,eval_kind,cv_splits,y_col_values):
	"高-H3緩和: 候補モデルのfold別R²のばらつき(標準偏差)を計算する（過学習/不安定性の\n    診断用、candidate_models.r2_std）。OOF評価(eval_kind=='oof')かつfold分割が2以上ある\n    場合のみ意味を持つ。非OOF(quickの単一train/val分割)はfold概念が1つしかなく分散を\n    計算できないため0.0を返す(=「参考値なし」として扱う)。";E=y_col_values;D=cv_splits;A=preds_arr
	if eval_kind!=_V or D is _A or A is _A:return _D
	A=np.asarray(A,dtype=float)
	if len(A)!=len(E):return _D
	B=[]
	for(F,C)in D:
		if len(C)<2:continue
		try:B.append(float(r2_score(E[C],A[C])))
		except Exception:continue
	if len(B)<2:return _D
	return float(np.std(B))
def _get_feat_cols(df,target_col):return[A for A in df.columns if A!=target_col and pd.api.types.is_numeric_dtype(df[A])and df[A].isna().mean()<MAX_MISS_RATE]
def _poly_top_feats_by_corr(df_source,feat_cols,target_col,top_k):
	'多項式Ridge用の top-k 特徴選択。中-M2: 以前は生スケール分散ベース(np.nanvar)だった\n    ため、mm単位の列がkm単位の列にスケールの違いだけで必ず勝ち、選択が実質無意味だった。\n    ターゲットとの絶対相関(ペアワイズ欠損除外)ベースに変更する。相関が計算できない列\n    (有効ペア2点未満・分散ゼロ等)はランキング対象から除外する。';E=top_k;D=df_source;B=feat_cols;F=pd.to_numeric(D[target_col],errors=_W).values.astype(float);A=[]
	for G in B:
		H=pd.to_numeric(D[G],errors=_W).values.astype(float);C=np.isfinite(H)&np.isfinite(F)
		if C.sum()<2:continue
		I,J=H[C],F[C]
		if np.std(I)<1e-12 or np.std(J)<1e-12:continue
		K=np.corrcoef(I,J)[0,1]
		if not np.isfinite(K):continue
		A.append((G,abs(float(K))))
	if not A:return list(B[:E])
	A.sort(key=lambda t:t[1],reverse=_B);L=set(A for(A,B)in A[:E]);return[A for A in B if A in L]
def _find_constant_and_duplicate_cols(df,feat_cols):
	'分散ゼロの定数列、および相関がほぼ1の重複列を検出する。';C=feat_cols;B=[]
	for A in C:
		D=df[A]
		if D.nunique(dropna=_B)<=1 or float(D.std(skipna=_B)or _D)<CONST_STD_EPS:B.append(A)
	E=[A for A in C if A not in B];F=[];G=[];H={A:df[A].fillna(df[A].median())for A in E}
	for A in E:
		I=_C
		for M in G:
			J,K=H[A],H[M]
			if J.std()>0 and K.std()>0:
				L=J.corr(K)
				if L is not _A and abs(L)>DUP_CORR_THRESH:I=_B;break
		if I:F.append(A)
		else:G.append(A)
	return B,F
def _split_es_holdout(dtr,seed_offset=0):
	'fold-train（またはquickのdf_train）内をさらに90/10に分割し、10%をLightGBMの\n    early stopping専用valとして返す。以前は評価対象のfold(OOF)やdf_val(quick)自身を\n    ESのvalとしても使い回しており、LGBMだけが評価データに直接フィットして系統的に\n    有利になりOOF/検証R²を楽観化させていた(高-H2)。ここで切り出す10%はbest_iteration\n    の決定にのみ使い、oof_preds/最終R²の算出には一切使わない。\n    Returns: (fit_df, es_df)。行数不足時は (dtr, None)（=ESなしで固定本数学習にフォールバック）。';A=dtr;B=len(A)
	if B<ES_MIN_TRAIN_ROWS:return A,_A
	E=np.random.RandomState(ES_SPLIT_SEED+seed_offset);D=E.permutation(B);C=max(1,int(round(B*ES_VAL_FRAC)))
	if B-C<10:return A,_A
	F,G=D[:C],D[C:];return A.iloc[G],A.iloc[F]
def _lgb_fit(sk_params,X,y,X_val=_A,y_val=_A,early_stopping=0):
	'lgb.train で Booster を学習して返す。n_estimators は num_boost_round に振り替える。';D=early_stopping;C=X_val;import lightgbm as A;B=dict(sk_params);B.setdefault('objective',_z);H=int(B.pop(_r,100));E=A.Dataset(X,label=y,free_raw_data=_C);F=_A;G=[A.log_evaluation(-1)]
	if C is not _A and D>0:F=[A.Dataset(C,label=y_val,reference=E)];G.append(A.early_stopping(D,verbose=_C))
	return A.train(B,E,num_boost_round=H,valid_sets=F,callbacks=G)
def _lgb_importance(bst,n_features):
	'Booster の split 重要度（LGBMRegressor.feature_importances_ 既定と同じ）。';B=n_features;A=np.asarray(bst.feature_importance(importance_type='split'),dtype=float)
	if len(A)<B:A=np.concatenate([A,np.zeros(B-len(A))])
	return A
def _lgbm_feature_screen(df_train,target_col,num_jobs=4):
	'軽量 LightGBM で重要度ゼロの特徴量を除外し、GP/MLP の次元を削減する。';D=target_col;B=df_train;A=_get_feat_cols(B,D)
	if len(A)<=LGBM_SCREEN_MIN_FEATS:return A
	try:
		E=B[A].median();F=B[A].fillna(E).values;G=B[D].values;H=_lgb_fit(dict(n_estimators=200,num_leaves=31,learning_rate=.1,verbosity=-1,n_jobs=num_jobs,force_col_wise=_B,min_child_samples=max(3,len(B)//30)),F,G);I=_lgb_importance(H,len(A));C=[A[B]for B in range(len(A))if I[B]>0]
		if len(C)<2:return A
		if len(C)<len(A):print(f"[Screen] LGBM probe: {len(A)}→{len(C)} 列に絞込み (GP/MLP用)",flush=_B)
		return C
	except Exception as J:print(f"[Screen] 特徴量スクリーニング失敗 → 全列使用: {J}",flush=_B);return A
def _build_derived_recipe(df_train,target_col,num_jobs=4):
	'ペア積・二乗・符号の派生特徴候補を生成し、LGBM 重要度で上位を選抜する。\n    選抜は既存の特徴量スクリーニングと同様に fold0 の学習側で行う。\n    Returns: [{"name", "op", "cols"}]  (op: mul / sq / sign)';N=num_jobs;M=target_col;A=df_train;B=_get_feat_cols(A,M)
	if len(B)<2:return[]
	try:
		R=A[B].median();E=A[B].fillna(R);O=A[M].values;H=B
		if len(B)>FE_MAX_RAW:S=_lgb_fit(dict(n_estimators=150,num_leaves=31,learning_rate=.1,verbosity=-1,n_jobs=N,force_col_wise=_B),E.values,O);T=np.argsort(_lgb_importance(S,len(B)))[::-1];H=[B[A]for A in T[:FE_MAX_RAW]]
		U=set(A.columns);F=[];I={}
		def J(name,op,cols,values):
			A=name
			if A in U or A in I:return
			B=np.asarray(values,dtype=float)
			if not np.isfinite(B).all()or B.std()<1e-12:return
			F.append({_G:A,_h:op,_i:cols});I[A]=B
		for(V,C)in enumerate(H):
			D=E[C].values
			for K in H[V+1:]:J(f"{C}*{K}",'mul',[C,K],D*E[K].values)
			J(f"{C}^2",'sq',[C],D*D)
			if(D>0).any()and(D<0).any():J(f"sign({C})",'sign',[C],np.sign(D))
		if not F:return[]
		W=pd.DataFrame(I,index=A.index);L=pd.concat([E,W],axis=1);X=_lgb_fit(dict(n_estimators=200,num_leaves=31,learning_rate=.05,verbosity=-1,n_jobs=N,force_col_wise=_B,min_child_samples=max(3,len(A)//30)),L.values,O);Y=pd.Series(_lgb_importance(X,L.shape[1]),index=list(L.columns));P=Y[[A[_G]for A in F]].sort_values(ascending=_C);Z=set(P[P>0].head(FE_TOP_K).index);G=[A for A in F if A[_G]in Z]
		if G:Q=[A[_G]for A in G];print(f"[FE] 自動特徴量 {len(G)} 本を採用: {', '.join(Q[:5])}{' …'if len(Q)>5 else''}",flush=_B)
		return G
	except Exception as a:print(f"[FE] 自動特徴量生成失敗 → スキップ: {a}",flush=_B);return[]
def _apply_derived(df,recipe):
	'派生特徴レシピを DataFrame に適用する。\n    ソース欠損・非有限は NaN として伝播し、後段の median 補完に委ねる。';F=recipe;A=df
	if not F:return A
	A=A.copy();G=pd.Series(np.nan,index=A.index)
	for B in F:
		C=B[_i];D=pd.to_numeric(A[C[0]],errors=_W)if C[0]in A.columns else G
		if B[_h]=='mul':H=pd.to_numeric(A[C[1]],errors=_W)if len(C)>1 and C[1]in A.columns else G;E=(D*H).values.astype(float)
		elif B[_h]=='sq':E=(D*D).values.astype(float)
		elif B[_h]=='sign':E=np.sign(D.values.astype(float))
		else:continue
		A[B[_G]]=np.where(np.isfinite(E),E,np.nan)
	return A
def _r2_interpretation(r2):
	"戻り値はキー(フロントのI18N['r2interp.'+key]で表示言語に翻訳される)。\n    テスト・ベンチマークからは参照されていないため直接キー化している。";A=r2
	if A>=.95:return'very_high'
	elif A>=.85:return'high'
	elif A>=.7:return'practical'
	elif A>=.5:return'moderate'
	elif A>=_D:return'insufficient'
	else:return'nonfunctional'
def _clean_model_files(model_dir,keep_type,keep_lgbm_bag=_C):
	C=keep_type;B=model_dir;F={_a:[_j],_X:[_s,_A0],_b:[_t],_Y:[_u],_e:['rf_model.txt','rf_meta.json'],_n:['xt_model.txt','xt_meta.json']}
	for(G,H)in F.items():
		if G!=C:
			for I in H:
				A=os.path.join(B,I)
				if os.path.exists(A):os.remove(A)
	if C!=_X and not keep_lgbm_bag:
		D=os.path.join(B,_AE)
		if os.path.exists(D):os.remove(D)
		E=0
		while _B:
			A=os.path.join(B,f"lgbm_model_fold{E}.txt")
			if not os.path.exists(A):break
			os.remove(A);E+=1
def _read_csv_with_encoding_fallback(csv_path):
	'UTF-8として妥当かをまずバイト列で検査し、無効なら Shift-JIS(cp932) として読む。\n    日本語Excelが既定で書き出すShift-JIS CSVがUTF-8として「�」化けしたまま\n    サイレントに学習が完走してしまうのを防ぐ(中-7)。';B=csv_path
	with open(B,_o)as C:D=C.read()
	try:
		try:D.decode(_R);A=pd.read_csv(B,encoding='utf-8-sig')
		except UnicodeDecodeError:print(f"[Python] CSVがUTF-8として不正 → Shift-JIS(cp932)として読み込みます",flush=_B);A=pd.read_csv(B,encoding='cp932')
	except pd.errors.EmptyDataError:_error_exit('csv_empty')
	except pd.errors.ParserError as E:_error_exit('csv_parse_failed',detail=str(E)[:200])
	A.columns=A.columns.str.strip();return A
def _resolve_and_validate_target(df,target_column_arg):
	'target 列の存在・数値性を検証し、NaN 行を除去する。\n    Returns: (df, target_column, n_target_na)。エラーは print + exit(1)。';C=target_column_arg;A=df
	if C:
		if C not in A.columns:_error_exit('target_col_not_found',col=C)
		B=C;print(f"[Python] ターゲット: 「{B}」",flush=_B)
	else:B=A.columns[-1];print(f"[Python] ターゲット自動判定: 「{B}」",flush=_B)
	D=A[B]
	if not pd.api.types.is_numeric_dtype(D):
		F=pd.to_numeric(D,errors=_W);G=F.isna()&D.notna()
		if G.any():I=str(D[G].iloc[0]);_error_exit('target_col_not_numeric',col=B,example=I)
		A=A.copy();A[B]=F
	H=int(np.isinf(A[B]).sum())
	if H>0:A=A.copy();A[B]=A[B].replace([np.inf,-np.inf],np.nan);print(f"[Python] ターゲットが±無限大の {H} 行を欠損として扱います",flush=_B)
	E=int(A[B].isna().sum())
	if E>0:A=A.dropna(subset=[B]).reset_index(drop=_B);print(f"[Python] ターゲット欠損 {E} 行を除外",flush=_B)
	if len(A)==0:_error_exit('target_all_na')
	if A[B].nunique()<=1:_error_exit('target_constant',col=B,value=float(A[B].iloc[0]))
	return A,B,E
def _detect_y_transform(y,y_full=_A):
	'skew判定は y（fold0-train等の部分集合で可）で行うが、log1p適用可否のmin>=0判定は\n    y_full（全行）で行う。負値行が別foldに落ちても log1p(負値)=NaN が学習ターゲットに\n    混入するのを防ぐため。y_full省略時はyそのものを使う（後方互換）。';B=y_full
	if B is _A:B=y
	from _light import skew as E;A=float(E(y));print(f"[YTransform] Y skewness={A:.3f}",flush=_B)
	if A>SKEW_THRESH and float(B.min())>=0:print(f"[YTransform] log1p 変換を適用（skewness={A:.2f} > {SKEW_THRESH}, min≥0）",flush=_B);return _k,{}
	if abs(A)>SKEW_THRESH:
		try:from _light import PowerTransformer as F;C=F(method=_v,standardize=_C);C.fit(y.reshape(-1,1));D=float(C.lambdas_[0]);print(f"[YTransform] Yeo-Johnson 変換を適用（skewness={A:.2f}, λ={D:.3f}）",flush=_B);return _l,{_p:D}
		except Exception as G:print(f"[YTransform] Yeo-Johnson 失敗 → 変換なし: {G}",flush=_B)
	return _M,{}
Y_TRANSFORM_CV_FOLDS=2
Y_WINSORIZE_CV_MARGIN=.003
def _select_y_transform_cv(df,target_col,cv_splits,num_jobs=4):
	"精度レバー3: skewヒューリスティックの代わりに、none/log1p/yeo_johnsonの3通りを\n    軽量LGBM(先頭Y_TRANSFORM_CV_FOLDS=2fold)で比較し、外部スケールでのOOF R²が\n    最良のものを採用する(じっくりモード専用。お急ぎモードは_detect_y_transformの\n    skewヒューリスティックを維持)。yeo_johnsonのlambdaはfold0-trainでfitする\n    (既存の_detect_y_transformと同じリーク防止方針)。\n    候補が1つしかない・全候補が失敗した場合は('none', {})にフォールバックする。";I=cv_splits;H=target_col;C=df;J=C[H].values;T=I[0][0];B=[(_M,{})]
	if float(np.min(J))>=0:B.append((_k,{}))
	try:from _light import PowerTransformer as U;O=U(method=_v,standardize=_C);O.fit(J[T].astype(float).reshape(-1,1));V=float(O.lambdas_[0]);B.append((_l,{_p:V}))
	except Exception as K:print(f"[YTransform-CV] Yeo-Johnson fit失敗 → 候補から除外: {K}",flush=_B)
	if len(B)==1:return B[0]
	E=_get_feat_cols(C,H)
	if not E:return B[0]
	W=min(Y_TRANSFORM_CV_FOLDS,len(I));A=_A
	for(D,L)in B:
		M=np.full(len(C),np.nan);P=_B
		for(X,Q)in I[:W]:
			F,Y=C.iloc[X],C.iloc[Q];R=F[E].median();Z=F[E].fillna(R).values;a=Y[E].fillna(R).values
			try:
				S=_apply_y_transform(F[H].values,D,L)
				if not np.all(np.isfinite(S)):raise ValueError('y_tr contains non-finite values')
				b=_lgb_fit(dict(n_estimators=200,num_leaves=31,learning_rate=.1,verbosity=-1,n_jobs=num_jobs,force_col_wise=_B,deterministic=_B,seed=42,min_child_samples=max(3,len(F)//30)),Z,S);M[Q]=_invert_y_transform(b.predict(a),D,L)
			except Exception as K:print(f"[YTransform-CV] {D}: 予選失敗 → 候補から除外 ({K})",flush=_B);P=_C;break
		if not P:continue
		N=np.isfinite(M)
		if N.sum()<2:continue
		try:G=float(r2_score(J[N],M[N]))
		except Exception:continue
		if not np.isfinite(G):continue
		print(f"[YTransform-CV] {D}: 予選R²={G:.4f}",flush=_B)
		if A is _A or G>A[0]:A=G,D,L
	if A is _A:print('[YTransform-CV] 全候補が失敗 → 変換なしにフォールバック',flush=_B);return _M,{}
	print(f"[YTransform-CV] 採用: {A[1]} (予選R²={A[0]:.4f})",flush=_B);return A[1],A[2]
def _apply_y_transform(y,transform,params):
	A=transform
	if A==_k:return np.log1p(y)
	if A==_l:from _light import PowerTransformer as C;B=C(method=_v,standardize=_C);B.lambdas_=np.array([params[_p]]);return B.transform(y.reshape(-1,1)).ravel()
	return y.copy()
def _invert_y_transform(y,transform,params):
	A=transform
	if A==_k:return np.expm1(y)
	if A==_l:from _light import PowerTransformer as C;B=C(method=_v,standardize=_C);B.lambdas_=np.array([params[_p]]);return B.inverse_transform(y.reshape(-1,1)).ravel()
	return y.copy()
def _fit_postprocess_params(preds,y_true,y_transform,y_raw_all,target_is_integer):
	'y_raw_all は winsorize 前の生 y（クリップ範囲が学習用加工に狭められないように）。';L='ignore';F=y_true;E=preds;B=y_raw_all;B=np.asarray(B,dtype=float);G,C=float(np.min(B)),float(np.max(B));H=C-G;I=H*Y_CLIP_MARGIN_FRAC if H>0 else max(abs(C),_E);M,N=G-I,C+I;J=_E
	if y_transform==_k and E is not _A and F is not _A:
		D=np.asarray(E,dtype=float);K=np.asarray(F,dtype=float)
		if len(D)==len(K)and len(D)>0:
			with np.errstate(divide=L,invalid=L):A=K/np.clip(D,1e-06,_A)
			A=A[np.isfinite(A)]
			if len(A)>0:J=float(np.clip(np.median(A),SMEAR_CLIP_RANGE[0],SMEAR_CLIP_RANGE[1]))
	return J,M,N,bool(target_is_integer)
def _apply_postprocess(preds,smear,y_clip_lo,y_clip_hi,round_output):
	A=preds;A=np.asarray(A,dtype=float)*smear;A=np.clip(A,y_clip_lo,y_clip_hi)
	if round_output:A=_round_half_away(A)
	return A
def _fit_y_winsorize_bounds(y_train,mult):A=y_train;B,C=np.percentile(A,25),np.percentile(A,75);D=C-B;return B-mult*D,C+mult*D
def _apply_y_winsorize(df,target_col,lo,hi):
	C=target_col;A=df;B=A[C].values;D=int(((B<lo)|(B>hi)).sum())
	if D>0:A=A.copy();A[C]=np.clip(B,lo,hi)
	return A,D
def _select_y_winsorize_cv(df,target_col,cv_splits,iqr_mult,num_jobs=4):
	'対策(2026-07): Y外れ値 winsorize を一律適用せず、clip/no-clip を LGBM 2-fold 予選で\n    「生y(=クリップ前)に対する OOF R²」で比較して決める。cpu_act 等の“正規の重い裾”を持つ\n    ターゲットでは一律クリップが裾の信号を破壊し test R² を大きく落とすため(実測: AutoGluon\n    比較ベンチ real_cpu で 0.96→0.78)。予選では各 fold の学習側 y にのみ候補のクリップを施し、\n    検証は必ず生yで採点する(=真の目的関数)。クリップが非クリップを Y_WINSORIZE_CV_MARGIN\n    以上上回った時のみクリップ採用(迷ったら非クリップ=信号温存を優先)。\n    戻り値: 採用すべき mult(float=クリップ実施)または None(クリップしない)。予選不能時は\n    従来動作(iqr_mult でクリップ)にフォールバック。';N=cv_splits;I=target_col;E=iqr_mult;D=df;C='clip';B='noclip';S=D[I].values.astype(float);F=_get_feat_cols(D,I)
	if not F:return E
	T=min(Y_TRANSFORM_CV_FOLDS,len(N));A={}
	for(J,O)in((B,_A),(C,E)):
		K=np.full(len(D),np.nan);P=_B
		for(U,Q)in N[:T]:
			G,V=D.iloc[U],D.iloc[Q];R=G[F].median();W=G[F].fillna(R).values;X=V[F].fillna(R).values;H=G[I].values.astype(float).copy()
			if O is not _A:Y,Z=_fit_y_winsorize_bounds(H,O);H=np.clip(H,Y,Z)
			try:a=_lgb_fit(dict(n_estimators=200,num_leaves=31,learning_rate=.1,verbosity=-1,n_jobs=num_jobs,force_col_wise=_B,deterministic=_B,seed=42,min_child_samples=max(3,len(G)//30)),W,H);K[Q]=a.predict(X)
			except Exception as b:print(f"[YWinsorize-CV] {J}: 予選失敗 → 除外 ({b})",flush=_B);P=_C;break
		if not P:continue
		L=np.isfinite(K)
		if L.sum()<2:continue
		try:M=float(r2_score(S[L],K[L]))
		except Exception:continue
		if np.isfinite(M):A[J]=M;print(f"[YWinsorize-CV] {J}: 予選R²(生y)={M:.4f}",flush=_B)
	if C not in A and B not in A:return E
	if B not in A:return E
	if C not in A:return
	if A[C]>A[B]+Y_WINSORIZE_CV_MARGIN:print(f"[YWinsorize-CV] 採用: クリップ (R² {A[C]:.4f} > 非クリップ {A[B]:.4f}+{Y_WINSORIZE_CV_MARGIN})",flush=_B);return E
	print(f"[YWinsorize-CV] 採用: 非クリップ (クリップ {A[C]:.4f} ≤ 非クリップ {A[B]:.4f}+{Y_WINSORIZE_CV_MARGIN}) — 正規の裾を温存",flush=_B)
CAT_ONEHOT_MAX_CARD=10
CAT_DROP_CARD_FRAC=.5
CAT_NUMERIC_COERCE_MIN=.9
CAT_TARGET_ENC_SMOOTH=2e1
CAT_NAN_SENTINEL='__NaN__'
CAT_DATETIME_COERCE_MIN=.9
_DATETIME_RE=re.compile('^(\\d{4})([-/])(\\d{2})\\2(\\d{2})(?:[ T]?(\\d{2}):(\\d{2})(?::(\\d{2}))?)?$')
_DATETIME_DAYS_IN_MONTH=31,29,31,30,31,30,31,31,30,31,30,31
_DATETIME_PART_NAMES='hour','dow','month','epoch_days'
def _days_from_civil(y,m,d):'Howard Hinnant の days_from_civil(パブリックドメイン、epoch=1970-01-01=0)。\n    整数演算のみで外部日付ライブラリ不要。C++/JS版と参照日テストで一致確認済み\n    (scratchpad dt_unittest.*、1970-01-01/閏年境界/負のepoch_daysを含め全一致)。';y=y-(1 if m<=2 else 0);B=(y if y>=0 else y-399)//400;A=y-B*400;C=(153*(m+(-3 if m>2 else 9))+2)//5+d-1;D=A*365+A//4-A//100+C;return B*146097+D-719468
def _weekday_from_days(days):'0=Mon..6=Sun(pandasのdt.dayofweekと同じ規約)。1970-01-01(Thu)がdays=0で\n    (0+3)%7=3=Thuになるよう+3オフセット。C++/JS版は`((days%7)+10)%7`(truncating\n    modulo対策の等価式、参照日テストで一致確認済み)を使う。';return(days+3)%7
def _parse_datetime_parts(s):
	's(生文字列、fillna+astype(str)済み)を解析し、成功なら\n    (hour, dow, month, epoch_days) の4-tuple(intのみ)、失敗ならNoneを返す。\n    検出率計算(90%閾値)と実際の値抽出の両方でこの関数だけを使う\n    (『唯一の真実の判定』、C++/JS/predict_template.pyへ一字一句移植する)。';A=_DATETIME_RE.match(s.strip())
	if not A:return
	F,B,C=int(A.group(1)),int(A.group(3)),int(A.group(4));D=int(A.group(5))if A.group(5)else 0;G=int(A.group(6))if A.group(6)else 0
	if not 1<=B<=12 or not 1<=C<=31 or not 0<=D<=23 or not 0<=G<=59:return
	if C>_DATETIME_DAYS_IN_MONTH[B-1]:return
	E=_days_from_civil(F,B,C);H=_weekday_from_days(E);return D,H,B,E
NUMERIC_KEY_MAX_CARD_FRAC=.05
NUMERIC_KEY_MAX_GROUP_FRAC=.5
NUMERIC_KEY_MIN_ROWS=30
NUMERIC_KEY_MAX_COLS=2
NUMERIC_KEY_SEP='\x1f'
NUMERIC_KEY_PREFIX='__numkey__'
def _canon_numeric_key_part(v):
	'合成キー1パーツの正規化。整数相当の値のみサポートする(非整数を含む列は\n    _detect_numeric_composite_keyで候補から除外済み)ことで、学習時(pandas由来の\n    float)と予測時(生CSV文字列をparse_numeric_field相当でパースしたfloat)の\n    両方で完全に同じ文字列になることを保証する(指数表記等の言語間フォーマット差異を\n    設計上回避)。C++/JS/predict_template.pyへ一字一句移植する。\n    実バグ2026-08(M4): 丸め規則がPython(banker\'s round)/JS(Math.round=half-up)/\n    C++(half-away-from-zero)で三者三様だった(例: 2.5→Py"2"/JS"3"/C++"3"、\n    -2.5→Py"-2"/JS"-2"/C++"-3")ため、既存の_round_half_away(half-away-from-zero、\n    round_output後処理と共通)で統一する。'
	if v is _A:return CAT_NAN_SENTINEL
	try:A=float(v)
	except(TypeError,ValueError):return CAT_NAN_SENTINEL
	if not np.isfinite(A)or abs(A)>=1e15:return CAT_NAN_SENTINEL
	return str(int(_round_half_away(A)))
def _detect_numeric_composite_key(df,target_col,exclude_cols=_A):
	'dfに残っている数値列(bool/カテゴリ処理済みで既に除外されたものは対象外)から、\n    単体で低カーディナリティかつ整数相当の値のみを持つものを集め、それらの組み合わせを\n    1本の合成キー文字列列としてdfに追加する。元の数値列は削除しない。\n    exclude_cols: one-hot indicator列(0.0/1.0の低カーディナリティ数値列で、そのままでは\n    合成キー候補の条件に合致してしまう)等、既に他のカテゴリエンコーダが生成した派生列を\n    候補から除外するための列名集合(呼び出し側が cat_onehot_specs/cat_datetime_specs の\n    feature_name を渡す)。\n    Returns: (df, key_col_name, source_cols) または候補なしなら (df, None, None)。';F=exclude_cols;A=df;D=len(A)
	if D<NUMERIC_KEY_MIN_ROWS:return A,_A,_A
	F=F or set();B=[]
	for C in A.columns:
		if C==target_col or C in F:continue
		if not pd.api.types.is_numeric_dtype(A[C])or pd.api.types.is_bool_dtype(A[C]):continue
		L=A[C];G=L.dropna()
		if len(G)==0:continue
		J=G.nunique()
		if J<2 or J/D>NUMERIC_KEY_MAX_CARD_FRAC:continue
		H=G.values.astype(float)
		if not np.all(H==np.round(H))or np.any(np.abs(H)>=1e15):continue
		B.append(C)
	if len(B)==0:return A,_A,_A
	if len(B)>NUMERIC_KEY_MAX_COLS:print(f"[CatEnc] 合成キー候補{B}が{NUMERIC_KEY_MAX_COLS}列を超えるため見送り(独立した低カーディナリティ属性の偶然の一致の疑い)",flush=_B);return A,_A,_A
	def M(row):return NUMERIC_KEY_SEP.join(_canon_numeric_key_part(row[A])for A in B)
	K=A[B].apply(M,axis=1);E=K.nunique()
	if E<2 or E/D>NUMERIC_KEY_MAX_GROUP_FRAC:return A,_A,_A
	I=NUMERIC_KEY_PREFIX+'_'.join(B);A=A.copy();A[I]=K;print(f"[CatEnc] 合成キー検出: {B} → {E}グループ(行数比{100*E/D:.1f}%) を{I}としてtarget encoding対象に追加",flush=_B);return A,I,B
def _prepare_categoricals(df,target_col):
	'object/bool dtype列を走査し、(df, onehot_specs, target_cols, dropped_cols,\n    datetime_specs) を返す。df は onehot・datetime_parts列(いずれもtarget非依存で\n    安全にグローバル適用可能)を実際に追加・元列削除したコピー。target_cols は\n    「まだ生の文字列(NaNは CAT_NAN_SENTINEL で埋め済み)のまま」残す列名リストで、\n    呼び出し側が split 後に _fit_target_encoders/_apply_target_encoders で\n    fold-aware に数値化する(target統計を使うため split 前にfitするとリークする)。\n    onehot_specs の要素: {"feature_name","source_col","class_value"}(生成された\n    indicator列名 → 元列・比較対象クラス値)。datetime_specs の要素:\n    {"feature_name","source_col","method":"datetime","part"}(1source_colにつき\n    hour/dow/month/epoch_daysの4エントリ)。';A=df;A=A.copy();S=[B for B in A.columns if B!=target_col and(pd.api.types.is_bool_dtype(A[B])or not pd.api.types.is_numeric_dtype(A[B]))];I,J,K,L=[],[],[],[];E=len(A);F=set(A.columns)
	for B in S:
		T=pd.api.types.is_bool_dtype(A[B])or pd.api.types.infer_dtype(A[B],skipna=_B)=='boolean'
		if not T:
			M=pd.to_numeric(A[B],errors=_W);N=float(M.notna().mean())if E>0 else _D
			if N>=CAT_NUMERIC_COERCE_MIN:A[B]=M;print(f"[CatEnc] {B}: {N*100:.0f}%が数値化可能 → 数値列として扱う",flush=_B);continue
		U=A[B].isna().values;V=A[B].astype(str).values;O=[_A if A else _parse_datetime_parts(B)for(A,B)in zip(U,V)];W=sum(1 for A in O if A is not _A);P=W/E if E>0 else _D
		if P>=CAT_DATETIME_COERCE_MIN:
			print(f"[CatEnc] {B}: {P*100:.0f}%が日時として解析可能 → datetime_parts(hour/dow/month/epoch_days)",flush=_B)
			for(X,Q)in enumerate(_DATETIME_PART_NAMES):
				C=f"{B}__{Q}"
				if C in F:print(f"[CatEnc] {B}: 生成列名 {C} が既存列と衝突 → このdatetimeパートをスキップ",flush=_B);continue
				L.append({_S:C,_P:B,_f:_A1,'part':Q});A[C]=[A[X]if A is not _A else np.nan for A in O];F.add(C)
			A=A.drop(columns=[B]);continue
		G=A[B].fillna(CAT_NAN_SENTINEL).astype(str);R=sorted(G.unique().tolist());D=len(R)
		if D>max(1,int(E*CAT_DROP_CARD_FRAC)):K.append(B);A=A.drop(columns=[B]);print(f"[CatEnc] {B}: カーディナリティ{D}が行数の{CAT_DROP_CARD_FRAC*100:.0f}%超 → 除外",flush=_B);continue
		if D<=CAT_ONEHOT_MAX_CARD:
			print(f"[CatEnc] {B}: {D}クラス → one-hot",flush=_B)
			for H in R:
				C=f"{B}=={H}"
				if C in F:print(f"[CatEnc] {B}: 生成列名 {C} が既存列と衝突 → このクラスをスキップ",flush=_B);continue
				I.append({_S:C,_P:B,_AF:H,_f:'onehot'});A[C]=(G==H).astype(np.float64);F.add(C)
			A=A.drop(columns=[B])
		else:print(f"[CatEnc] {B}: {D}クラス → fold内target encoding(後段でfit)",flush=_B);A[B]=G;J.append(B)
	return A,I,J,K,L
def _fit_target_encoders(df_fit,target_col,target_cols,smoothing=CAT_TARGET_ENC_SMOOTH):
	'target_cols(生文字列、CAT_NAN_SENTINEL埋め済み)を df_fit(必ず学習側のみ。\n    検証/評価側の行を含めるとOOFがリークする)でfitする。スムージング:\n    encoded = (count*cat_mean + smoothing*global_mean) / (count+smoothing)。\n    未知カテゴリ・fit時に見なかったNaNは global_mean にフォールバックする。';F=smoothing;E=target_cols;B=df_fit;C=[]
	if not E:return C
	D=pd.to_numeric(B[target_col],errors=_W).astype(float);G=float(np.nanmean(D.values))if len(D)else _D
	for A in E:
		if A not in B.columns:continue
		J=B[A].astype(str);H=D.groupby(J);I=H.count();K=H.mean();L=(I*K+F*G)/(I+F);M={str(A):float(B)for(A,B)in L.items()};C.append({_S:A,_P:A,_f:_A3,'map':M,_A2:G})
	return C
def _apply_target_encoders(df,specs):
	D=specs;A=df
	if not D:return A
	A=A.copy()
	for B in D:
		C=B[_P]
		if C not in A.columns:continue
		E,F=B['map'],B[_A2];A[C]=A[C].astype(str).map(lambda v,m=E,d=F:m.get(v,d)).astype(np.float64)
	return A
def _compute_x_clip(df,feat_cols):
	E,F=X_CLIP_PCTILE;B={}
	for A in feat_cols:
		C=float(df[A].quantile(E/1e2));D=float(df[A].quantile(F/1e2))
		if C<D:B[A]=[C,D]
	return B
def _apply_x_clip(df,bounds):
	A=df;A=A.copy()
	for(B,(C,D))in bounds.items():
		if B in A.columns:A[B]=A[B].clip(lower=C,upper=D)
	return A
def _make_binned_splits(df,target_col,n_splits=5,seed=42):
	'対策(2026-07 第2弾・真因⑤)として重複行グループ化CVを試作したが、40問ベンチの\n    実測でreal_winequalityのtest R²を悪化させ(重複行グループ化でfold構成が変わり、\n    Blendの重みfit結果自体が変化して汎化が悪化。ridgeスタッカー選択とは無関係と切り分け\n    済み)、効果不明・複雑性増のため撤回し元の実装に戻した。重複行対策は今後の課題。';A=n_splits;B=len(df);A=max(2,min(A,B//2));E=min(A,max(2,B//10));F=df[target_col].values
	try:C=pd.qcut(F,q=E,labels=_C,duplicates='drop');C=pd.Series(C).fillna(0).astype(int).values
	except Exception:C=pd.cut(pd.Series(F),bins=E,labels=_C).fillna(0).astype(int).values
	D=np.bincount(C);H=int(D[D>0].min())if(D>0).any()else 0;G=min(A,H)
	if G>=2:I=StratifiedKFold(n_splits=G,shuffle=_B,random_state=seed);return list(I.split(np.zeros(B),C))
	J=KFold(n_splits=max(2,min(A,B)),shuffle=_B,random_state=seed);return list(J.split(np.zeros(B)))
def _fold_frame(df_full,df_all_per_fold,fold_idx,tr_idx,target_col,base_feat_cols,screen_cols_per_fold=_A):
	'高-H1: OOF fold毎に「そのfoldの検証行を一切見ずに」fitしたFE/スクリーニング済み\n    データと特徴列を返す。df_all_per_fold が None の場合は従来通り共有 df_full/base_feat_cols\n    を使う(FE/screeningがそもそも無効な経路、または非OOF経路との後方互換)。\n    screen_cols_per_fold が与えられる場合(GP/MLP)は特徴列をそのfold専用のスクリーニング結果に\n    差し替える。それ以外(Linear/LGBM/RF/XT)は df_fold から都度 _get_feat_cols で再計算する\n    (fold毎にFE由来の派生列セットが異なり得るため)。';D=screen_cols_per_fold;C=fold_idx;B=df_all_per_fold
	if B is _A:return df_full,base_feat_cols
	A=B[C]
	if D is not _A:return A,D[C]
	return A,_get_feat_cols(A.iloc[tr_idx],target_col)
def _try_linear(df_train,df_val,target_col,model_dir,y_transform=_M,y_params={},df_all=_A,use_oof=_C,splits=_A,df_all_per_fold=_A):
	AC=df_all_per_fold;r=df_all;q=df_val;l=model_dir;Y=splits;K=df_train;G=y_params;F=y_transform;B=target_col
	try:
		import pickle as m;from _light import RidgeCV as V,RobustScaler as W,PolynomialFeatures as s;A=_get_feat_cols(K,B)
		if not A:return _A,[],_A,_A,_A
		N=K[A].median();t=len(K)<=POLY_MAX_ROWS and len(A)<=POLY_MAX_FEATS;AD=use_oof and r is not _A and Y is not _A;u=[.001,.01,.1,_E,1e1,1e2,1e3];v=[.01,.1,_E,1e1,1e2,1e3,1e4,1e5]
		if AD and not t:
			D=r;w=len(Y);P=np.zeros(len(D));n=D[A].median();print(f"[Linear] K-Fold OOF ({w} fold)...",flush=_B)
			for(x,(Z,a))in enumerate(Y):b,X=_fold_frame(D,AC,x,Z,B,A);O=b.iloc[Z];y=b.iloc[a];c=O[X].median();z=O[X].fillna(c).values;A0=_apply_y_transform(O[B].values,F,G);A1=y[X].fillna(c).values;d=W();e=V(alphas=u);e.fit(d.fit_transform(z),A0);A2=e.predict(d.transform(A1));P[a]=_invert_y_transform(A2,F,G)
			f=float(r2_score(D[B].values,P));print(f"[Linear] OOF R²={f:.4f}",flush=_B);g=D[A].fillna(n).values;A3=_apply_y_transform(D[B].values,F,G);A4=W();h=V(alphas=u);h.fit(A4.fit_transform(g),A3);A5=h.predict(A4.transform(g));A6=_invert_y_transform(A5,F,G);M=float(r2_score(D[B].values,A6));I=np.abs(h.coef_);Q=max(I.sum(),1e-09);R=sorted([{_G:A[B],_H:round(float(I[B]/Q*100),1)}for B in range(len(A))],key=lambda x:x[_H],reverse=_B);J={A:float(n[A])if not np.isnan(float(n[A]))else _D for A in A}
			with open(os.path.join(l,_j),_T)as S:m.dump({_Z:h,_U:A4,_I:A,_c:B,_q:_C,_F:J},S)
			print(f"[Linear] alpha={h.alpha_:.4g}",flush=_B);T={_K:_V,_N:A,_F:J,_Q:_B,_J:round(M,4)};return round(f,4),R,_a,P,T
		if AD and t:
			D=r;w=len(Y);n=D[A].median();A7=min(len(A),POLY_MAX_FEATS);E=_poly_top_feats_by_corr(D,A,B,A7);A8=D[E].median();P=np.zeros(len(D));print(f"[Linear] poly K-Fold OOF ({w} fold)...",flush=_B)
			for(x,(Z,a))in enumerate(Y):b,X=_fold_frame(D,AC,x,Z,B,A);O=b.iloc[Z];y=b.iloc[a];AH=min(len(X),POLY_MAX_FEATS);A9=_poly_top_feats_by_corr(O,X,B,AH);c=O[A9].median();z=O[A9].fillna(c).values;A0=_apply_y_transform(O[B].values,F,G);A1=y[A9].fillna(c).values;d=W();AE=s(degree=2,include_bias=_C,interaction_only=_C);e=V(alphas=v);e.fit(AE.fit_transform(d.fit_transform(z)),A0);A2=e.predict(AE.transform(d.transform(A1)));P[a]=_invert_y_transform(A2,F,G)
			f=float(r2_score(D[B].values,P));print(f"[Linear] poly OOF R²={f:.4f}",flush=_B);g=D[E].fillna(A8).values;A3=_apply_y_transform(D[B].values,F,G);H=W();L=s(degree=2,include_bias=_C,interaction_only=_C);C=V(alphas=v);C.fit(L.fit_transform(H.fit_transform(g)),A3);A5=C.predict(L.transform(H.transform(g)));A6=_invert_y_transform(A5,F,G);M=float(r2_score(D[B].values,A6));i=L.get_feature_names_out(E);I=np.abs(C.coef_);Q=max(I.sum(),1e-09);R=sorted([{_G:str(i[A]),_H:round(float(I[A]/Q*100),1)}for A in range(len(i))],key=lambda x:x[_H],reverse=_B);J={A:float(A8[A])if not np.isnan(float(A8[A]))else _D for A in E}
			with open(os.path.join(l,_j),_T)as S:m.dump({_Z:C,_U:H,'poly':L,_I:E,_c:B,_q:_B,_F:J},S)
			print(f"[Linear] poly alpha={C.alpha_:.3g}",flush=_B);T={_K:_V,_N:E,_F:J,_Q:_B,_J:round(M,4)};return round(f,4),R,_a,P,T
		AI=K[A].fillna(N).values;AF=_apply_y_transform(K[B].values,F,G);o=q if q is not _A else K;p=_m if q is not _A else _L
		if t:
			A7=min(len(A),POLY_MAX_FEATS);E=_poly_top_feats_by_corr(K,A,B,A7);AG=K[E].fillna(N[E]).values;H=W();L=s(degree=2,include_bias=_C,interaction_only=_C);j=L.fit_transform(H.fit_transform(AG));C=V(alphas=v);C.fit(j,AF);AJ=o[E].fillna(N[E]).values;AA=C.predict(L.transform(H.transform(AJ)));k=_invert_y_transform(AA,F,G);U=float(r2_score(o[B].values,k))
			if p==_L:M=U
			else:AB=C.predict(L.transform(H.transform(AG)));M=float(r2_score(K[B].values,_invert_y_transform(AB,F,G)))
			i=L.get_feature_names_out(E);I=np.abs(C.coef_);Q=max(I.sum(),1e-09);R=sorted([{_G:str(i[A]),_H:round(float(I[A]/Q*100),1)}for A in range(len(i))],key=lambda x:x[_H],reverse=_B);J={A:float(N[A])if not np.isnan(float(N[A]))else _D for A in E}
			with open(os.path.join(l,_j),_T)as S:m.dump({_Z:C,_U:H,'poly':L,_I:E,_c:B,_q:_B,_F:J},S)
			AK=j.shape[1];print(f"[Linear] poly R²={U:.4f}  α={C.alpha_:.3g}  poly_feats={AK}",flush=_B);T={_K:p,_N:E,_F:J,_Q:_B,_J:round(M,4)};return round(U,4),R,_a,k,T
		H=W();j=H.fit_transform(AI);C=V(alphas=u);C.fit(j,AF);AL=H.transform(o[A].fillna(N).values);AA=C.predict(AL);k=_invert_y_transform(AA,F,G);U=float(r2_score(o[B].values,k))
		if p==_L:M=U
		else:AB=C.predict(j);M=float(r2_score(K[B].values,_invert_y_transform(AB,F,G)))
		I=np.abs(C.coef_);Q=max(I.sum(),1e-09);R=sorted([{_G:A[B],_H:round(float(I[B]/Q*100),1)}for B in range(len(A))],key=lambda x:x[_H],reverse=_B);J={A:float(N[A])if not np.isnan(float(N[A]))else _D for A in A}
		with open(os.path.join(l,_j),_T)as S:m.dump({_Z:C,_U:H,_I:A,_c:B,_q:_C,_F:J},S)
		print(f"[Linear] R²={U:.4f}  alpha={C.alpha_:.4g}",flush=_B);T={_K:p,_N:A,_F:J,_Q:_B,_J:round(M,4)};return round(U,4),R,_a,k,T
	except Exception as AM:print(f"[Linear] 失敗: {AM}",flush=_B);return _A,[],_A,_A,_A
def _try_lgbm_steps(df_train,df_val,target_col,model_dir,use_grid=_C,use_oof=_C,y_transform=_M,y_params={},df_all=_A,num_jobs=4,splits=_A,prog=_A,df_all_per_fold=_A):
	'本体はジェネレータ。予選/本戦の1候補評価ごとにyield(値なし)し、_try_lgbm(同期)/\n    _try_lgbm_async(Pyodide向け非同期)が呼び出し方だけを変えて共有する(計算ロジックの複製を避ける)。\n    戻り値は StopIteration.value (関数末尾の return がそのまま伝播する)。';A3='learning_rate';m=df_all;X=df_val;R=use_grid;N=splits;M=model_dir;G=df_train;E=y_params;D=y_transform;B=target_col
	try:
		import lightgbm as AO;A=_get_feat_cols(G,B)
		if not A:return _A,[],_A,_A,_A
		A4=len(G);Y=dict(objective=_z,metric='rmse',verbosity=-1,n_jobs=num_jobs,force_col_wise=_B,deterministic=_B,min_child_samples=max(3,A4//30),subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_alpha=.1,reg_lambda=_E)
		def n(medians_dict):
			with open(os.path.join(M,_A0),'w',encoding=_R)as B:json.dump({_I:A,_F:medians_dict},B,ensure_ascii=_C)
		if use_oof and m is not _A and N is not _A:
			C=m;Z=len(N);H=_lgbm_search_candidates(len(C))if R else[LGBM_PARAM_QUICK]
			def o(param_override,fold_idxs):
				F=dict(Y);F.update(param_override);M=np.zeros(len(C));O=np.zeros(len(C),dtype=bool);P=[]
				for H in fold_idxs:
					Q,I=N[H];R,G=_fold_frame(C,df_all_per_fold,H,Q,B,A);S=R.iloc[Q];W=R.iloc[I];J=S[G].median();T,K=_split_es_holdout(S,seed_offset=H);U=T[G].fillna(J).values;V=_apply_y_transform(T[B].values,D,E);X=W[G].fillna(J).values
					if K is not _A:Z=K[G].fillna(J).values;a=_apply_y_transform(K[B].values,D,E);L=_lgb_fit(F,U,V,Z,a,early_stopping=50)
					else:L=_lgb_fit(F,U,V)
					b=L.predict(X);M[I]=_invert_y_transform(b,D,E);O[I]=_B;P.append(L.best_iteration or int(F.get(_r,100)))
				return M,O,P
			if R and len(H)>1 and Z>LGBM_HALVING_FOLDS:
				print(f"[LightGBM] ランダムサーチ {len(H)} 候補を予選 ({LGBM_HALVING_FOLDS}fold)...",flush=_B);p=[];q=len(H)
				for(O,F)in enumerate(H,1):
					if prog is not _A:r,A5=prog;_emit_progress(r+(A5-r)*.8*O/q,'lgbm_search',O,q)
					A6,s,t=o(F,range(LGBM_HALVING_FOLDS));a=float(r2_score(C[B].values[s],A6[s]));print(f"  予選{O}/{len(H)}: R²={a:.4f} (leaves={F[_A4]}, lr={F[A3]})",flush=_B);p.append(a if np.isfinite(a)else-np.inf);yield
				A7=np.argsort(p)[::-1][:LGBM_FINALISTS];S=[H[A]for A in A7];print(f"[LightGBM] 予選通過 {len(S)} 候補 → 全{Z}foldで本戦",flush=_B)
			else:S=H
			u=-np.inf;v=_A
			for(O,F)in enumerate(S,1):
				if R:print(f"[LightGBM] 候補{O}/{len(S)}: num_leaves={F[_A4]}, lr={F[A3]}, n={F[_r]}",flush=_B)
				T,t,U=o(F,range(Z));I=float(r2_score(C[B].values,T))
				if R:print(f"  OOF R²={I:.4f}  avg_iter={int(np.mean(U))}",flush=_B)
				if I>u:u=I;v=dict(Y,**F),U,T,I
				yield
			P,U,T,I=v;print(f"[LightGBM] 最良パラメータ R²={I:.4f}",flush=_B);A8=max(50,int(np.mean(U)*1.15));b=dict(P);b[_r]=A8;V=C[A].median();c=C[A].fillna(V).values;A9=_apply_y_transform(C[B].values,D,E);J={A:float(V[A])if not np.isnan(float(V[A]))else _D for A in A};n(J);d=len(N);e=[]
			for(w,t)in N:AA=C[A].iloc[w].fillna(V).values;AB=_apply_y_transform(C[B].values[w],D,E);e.append(_lgb_fit(b,AA,AB))
			for(AC,AD)in enumerate(e):AD.save_model(os.path.join(M,f"lgbm_model_fold{AC}.txt"))
			with open(os.path.join(M,_AE),'w',encoding=_R)as AE:json.dump({'n_folds':d,_I:A,_F:J},AE,ensure_ascii=_C)
			AF=np.mean([A.predict(c)for A in e],axis=0);x=float(r2_score(C[B].values,_invert_y_transform(AF,D,E)));f=_lgb_fit(b,c,A9);f.save_model(os.path.join(M,_s));g=f.predict(c);W=float(r2_score(C[B].values,_invert_y_transform(g,D,E)));Q=_lgb_importance(f,len(A));h=max(Q.sum(),_E);i=sorted([{_G:A[B],_H:round(float(Q[B]/h*100),1)}for B in range(len(A))],key=lambda x:x[_H],reverse=_B);j={_K:_V,_N:A,_F:J,_Q:_B,_J:round(W,4),_A5:d if d>=2 else _A,'bag_train_r2':round(x,4)if x is not _A else _A};return round(I,4),i,_X,T,j
		K=G[A].median();y=G[A].fillna(K).values;AG=_apply_y_transform(G[B].values,D,E);P=dict(Y);P.update(LGBM_PARAM_QUICK);z,k=_split_es_holdout(G)
		if k is not _A:AH=z[A].fillna(K).values;AI=_apply_y_transform(z[B].values,D,E);AJ=k[A].fillna(K).values;AK=_apply_y_transform(k[B].values,D,E);L=_lgb_fit(P,AH,AI,AJ,AK,early_stopping=30)
		else:L=_lgb_fit(P,y,AG)
		A0=X if X is not _A else G;A1=_m if X is not _A else _L;AL=A0[A].fillna(K).values;AM=L.predict(AL);A2=_invert_y_transform(AM,D,E);l=float(r2_score(A0[B].values,A2))
		if A1==_L:W=l
		else:g=L.predict(y);W=float(r2_score(G[B].values,_invert_y_transform(g,D,E)))
		Q=_lgb_importance(L,len(A));h=max(Q.sum(),_E);i=sorted([{_G:A[B],_H:round(float(Q[B]/h*100),1)}for B in range(len(A))],key=lambda x:x[_H],reverse=_B);L.save_model(os.path.join(M,_s));J={A:float(K[A])if not np.isnan(float(K[A]))else _D for A in A};n(J);print(f"[LightGBM] R²={l:.4f}  trees={L.num_trees()}",flush=_B);j={_K:A1,_N:A,_F:J,_Q:_B,_J:round(W,4)};return round(l,4),i,_X,A2,j
	except Exception as AN:print(f"[LightGBM] 失敗: {AN}",flush=_B);return _A,[],_A,_A,_A
def _try_lgbm(*A,**B):
	'同期版: _try_lgbm_steps を最後まで一気に回すだけの薄いラッパー(exe版が使う)。';C=_try_lgbm_steps(*A,**B)
	try:
		while _B:next(C)
	except StopIteration as D:return D.value
async def _try_lgbm_async(*A,**B):
	'非同期版: _try_lgbm_steps を1ステップずつ回し、その都度ブラウザに制御を返す(Pyodide向け)。';C=_try_lgbm_steps(*A,**B)
	try:
		while _B:next(C);await _maybe_yield()
	except StopIteration as D:return D.value
def _scipy_optimize_gp_hyperparams(X_tr_s,y_train,d,x0=_A,max_iter=200):
	'ARD-RBF + White カーネルの超パラメータを自前 L-BFGS で最尤推定する。\n    NLL/解析勾配・オプティマイザとも _light（numpy のみ）を使用（scipy 非依存）。\n    Returns: (ls_opt, sv_opt, nv_opt, nll_value)\n    ';A=y_train;from _light import gp_nll_and_grad as C,minimize_lbfgs as D;E=float(A.mean());F=max(float(A.std()),1e-06);G=(A-E)/F
	if x0 is _A:x0=np.zeros(d+2);x0[d+1]=-3.
	B,H=D(lambda p:C(p,X_tr_s,G,d),x0,max_iter=max_iter);I=np.exp(B[:d]).clip(.001,1e2);J=float(np.exp(B[d]).clip(.0001,1e4));K=float(np.exp(B[d+1]).clip(1e-06,1e1));return I,J,K,H
def _gp_feature_importance(gp,ls_opt,feat_cols,n_features):
	B=n_features;A=ls_opt
	if A is _A:
		try:A=np.atleast_1d(np.array(gp.length_scale,dtype=float))
		except Exception:A=_A
	if A is _A or len(A)!=B:return[]
	C=_E/np.clip(A,1e-09,_A);D=C/C.sum();return sorted([{_G:feat_cols[A],_H:round(float(D[A])*100,1)}for A in range(B)],key=lambda x:x[_H],reverse=_B)
def _try_gp_steps(df_train,df_val,target_col,model_dir,use_grid=_C,use_oof=_C,y_transform=_M,y_params={},df_all=_A,feat_cols_override=_A,splits=_A,df_all_per_fold=_A,screen_cols_per_fold=_A):
	'本体はジェネレータ(詳細は _try_lgbm_steps のdocstring参照)。fold単位でyield(値なし)する。';W=feat_cols_override;V=df_all;U=use_grid;T=model_dir;L=splits;K=df_val;H=y_params;G=y_transform;C=df_train;B=target_col
	try:
		import pickle as X;from _light import StandardScaler as n,LightGP as Y;A=W if W else _get_feat_cols(C,B)
		if not A:return _A,[],_A,_A,_A
		Z=len(A);o=GP_RESTARTS_THOROUGH if U else GP_RESTARTS_QUICK
		def M(dtr,dva_or_none,feat_cols_local=_A):
			Q=feat_cols_local;P=dva_or_none;J=dtr;D=Q if Q is not _A else A;C=len(D);E=J[D].median();K=J[D].fillna(E).values;R=J[B].values;S=len(K)
			if S>GP_MAX_TRAIN:T=np.random.RandomState(42).choice(S,GP_MAX_TRAIN,replace=_C);L,U=K[T],R[T]
			else:L,U=K,R
			V=_apply_y_transform(U,G,H);M=n();W=M.fit_transform(L);F=X=Z=_A
			if len(L)>=SCIPY_GP_MIN_ROWS:
				a=float('inf');N=_A
				for b in range(o):
					O=np.zeros(C+2)
					if b>0:O[:C]=np.random.RandomState(1000+b).randn(C)*.5
					O[C+1]=-3.
					try:
						e,f,g,c=_scipy_optimize_gp_hyperparams(W,V,C,x0=O)
						if c<a:a=c;N=e,f,g
					except Exception:pass
				if N is not _A:F,X,Z=N
			if F is not _A:I=Y(length_scale=F,sigma_var=X,noise_var=Z)
			else:I=Y(length_scale=np.ones(C),sigma_var=_E,noise_var=.01)
			I.fit(W,V);d=_A
			if P is not _A:h=P[D].fillna(E).values;i=I.predict(M.transform(h));d=_invert_y_transform(i,G,H)
			j={A:float(E[A])if not np.isnan(float(E[A]))else _D for A in D};return I,M,F,d,j
		if use_oof and V is not _A and L is not _A:
			D=V;p=len(L);N=np.zeros(len(D))
			for(a,(b,c))in enumerate(L):
				d,q=_fold_frame(D,df_all_per_fold,a,b,B,A,screen_cols_per_fold);r=d.iloc[b];e=d.iloc[c];I,I,I,f,I=M(r,e,feat_cols_local=q);N[c]=f
				if U:s=r2_score(e[B].values,f);print(f"  [GP] fold {a+1}/{p}: R²={s:.4f}",flush=_B)
				yield
			g=float(r2_score(D[B].values,N));print(f"[GP] OOF R²={g:.4f}",flush=_B);h,t,u,i,E=M(D,D);F=float(r2_score(D[B].values,i))if i is not _A else _A;O=_gp_feature_importance(h,u,A,Z)
			with open(os.path.join(T,_t),_T)as P:X.dump({_Z:h,_U:t,_I:A,_c:B,_F:E},P)
			Q={_K:_V,_N:A,_F:E,_Q:_B,_J:round(F,4)if F is not _A else _A};return round(g,4),O,_b,N,Q
		j=K if K is not _A else C;k=_m if K is not _A else _L;J,R,v,I,E=M(C,_A);l=C[A].median();w=j[A].fillna(l).values;x=J.predict(R.transform(w));m=_invert_y_transform(x,G,H);S=float(r2_score(j[B].values,m))
		if k==_L:F=S
		else:y=C[A].fillna(l).values;z=J.predict(R.transform(y));F=float(r2_score(C[B].values,_invert_y_transform(z,G,H)))
		O=_gp_feature_importance(J,v,A,Z)
		with open(os.path.join(T,_t),_T)as P:X.dump({_Z:J,_U:R,_I:A,_c:B,_F:E},P)
		print(f"[GP] R²={S:.4f}",flush=_B);Q={_K:k,_N:A,_F:E,_Q:_B,_J:round(F,4)};return round(S,4),O,_b,m,Q
	except Exception as A0:print(f"[GP] 失敗: {A0}",flush=_B);return _A,[],_A,_A,_A
def _try_gp(*A,**B):
	'同期版の薄いラッパー(exe版が使う)。';C=_try_gp_steps(*A,**B)
	try:
		while _B:next(C)
	except StopIteration as D:return D.value
async def _try_gp_async(*A,**B):
	'非同期版の薄いラッパー(Pyodide向け)。';C=_try_gp_steps(*A,**B)
	try:
		while _B:next(C);await _maybe_yield()
	except StopIteration as D:return D.value
def _mlp_feature_importance(pipeline,eval_df,feat_cols,target_col,y_transform,y_params):
	D=eval_df;A=feat_cols;from _light import permutation_importance as G;H=D[A].median();I=D[A].fillna(H).values;J=D[target_col].values
	class K:
		def __init__(A,pipe,t,p):A.pipe=pipe;A.t=t;A.p=p
		def fit(A,X,y=_A):return A
		def predict(A,X):return _invert_y_transform(A.pipe.predict(X),A.t,A.p)
		def score(A,X,y):return r2_score(y,A.predict(X))
	L=K(pipeline,y_transform,y_params);B,E=I,J
	if len(B)>100:F=np.random.RandomState(42).choice(len(B),100,replace=_C);B,E=B[F],E[F]
	try:M=G(L,B,E,n_repeats=5,random_state=42);C=M.importances_mean
	except Exception:C=np.zeros(len(A))
	N=max(float(C[C>0].sum()),1e-09);return sorted([{_G:A[B],_H:round(float(C[B])/N*100,1)if C[B]>0 else _D}for B in range(len(A))],key=lambda x:x[_H],reverse=_B)
def _try_mlp_steps(df_train,df_val,target_col,model_dir,use_grid=_C,use_oof=_C,y_transform=_M,y_params={},df_all=_A,feat_cols_override=_A,splits=_A,df_all_per_fold=_A,screen_cols_per_fold=_A):
	'本体はジェネレータ(詳細は _try_lgbm_steps のdocstring参照)。fold単位でyield(値なし)する。';A7='n_iter_';t=screen_cols_per_fold;s=df_all_per_fold;r=feat_cols_override;q=df_all;p=model_dir;o='single_layer';Z=df_val;Y='alpha';T=splits;K=y_params;J=y_transform;G=use_grid;F=df_train;A=target_col
	try:
		import pickle as u;from _light import StandardScaler as A8,LightMLP as A9,LightPipeline as AA;B=r if r else _get_feat_cols(F,A)
		if not B or len(F)<MLP_MIN_ROWS:return _A,[],_A,_A,_A
		v=len(B);AB=min(256,max(32,v*4));AC=min(128,max(16,v*2));E=_mlp_search_candidates()if G else[MLP_PARAM_QUICK]
		def AD(param_spec,n_rows_local):
			A=param_spec;F=A.get(Y,.0001);H=A.get(o,_C);I=A.get('extra_layer',_C);D=A.get('width',_E);B=max(8,int(AB*D));E=max(4,int(AC*D))
			if H:C=B,
			elif I:C=B,E,B
			else:C=B,E
			J=1500 if G else 600;K=A9(hidden_layer_sizes=C,alpha=F,max_iter=J,learning_rate_init=.01,random_state=42);L=AA([(_U,A8()),(_Y,K)]);return L,C
		def U(dtr,dva_or_none,param_spec,feat_cols_local=_A):
			H=feat_cols_local;G=dva_or_none;C=dtr;D=H if H is not _A else B;E=C[D].median();L=C[D].fillna(E).values;M=_apply_y_transform(C[A].values,J,K);F,N=AD(param_spec,len(C));F.fit(L,M);I=_A
			if G is not _A:O=G[D].fillna(E).values;P=F.predict(O);I=_invert_y_transform(P,J,K)
			Q={A:float(E[A])if not np.isnan(float(E[A]))else _D for A in D};return F,N,I,Q
		if use_oof and q is not _A and T is not _A:
			C=q;w=len(T);L=E
			if G and len(E)>MLP_FINALISTS and w>MLP_HALVING_FOLDS:
				print(f"[MLP] ランダムサーチ {len(E)} 候補を予選 ({MLP_HALVING_FOLDS}fold)...",flush=_B);x=[]
				for(M,D)in enumerate(E,1):
					y=np.zeros(len(C));a=np.zeros(len(C),dtype=bool)
					try:
						for(AE,(N,H))in enumerate(T[:MLP_HALVING_FOLDS]):O,b=_fold_frame(C,s,AE,N,A,B,t);c=O.iloc[N];d=O.iloc[H];P,P,e,P=U(c,d,D,feat_cols_local=b);y[H]=e;a[H]=_B;yield
						V=float(r2_score(C[A].values[a],y[a]))
					except Exception:V=-np.inf
					print(f"  予選{M}/{len(E)}: R²={V:.4f} (alpha={D.get(Y):.2g})",flush=_B);x.append(V if np.isfinite(V)else-np.inf)
				AF=np.argsort(x)[::-1][:MLP_FINALISTS];L=[E[A]for A in AF];print(f"[MLP] 予選通過 {len(L)} 候補 → 全{w}foldで本戦",flush=_B)
			W=-np.inf;z=L[0];A0=_A
			for(M,D)in enumerate(L,1):
				if G:print(f"[MLP] 候補{M}/{len(L)}: alpha={D.get(Y):.2g}, single_layer={D.get(o,_C)}",flush=_B)
				f=np.zeros(len(C))
				for(AG,(N,H))in enumerate(T):O,b=_fold_frame(C,s,AG,N,A,B,t);c=O.iloc[N];d=O.iloc[H];P,P,e,P=U(c,d,D,feat_cols_local=b);f[H]=e;yield
				g=float(r2_score(C[A].values,f))
				if G:print(f"  OOF R²={g:.4f}",flush=_B)
				if g>W:W=g;z=D;A0=f
			Q,h,A1,X=U(C,C,z);R=float(r2_score(C[A].values,A1))if A1 is not _A else _A;i=_mlp_feature_importance(Q,C,B,A,J,K)
			with open(os.path.join(p,_u),_T)as j:u.dump({_A6:Q,_I:B,_c:A,_F:X},j)
			k=getattr(Q[_Y],A7,'?');print(f"[MLP] OOF R²={W:.4f}  hidden={h}  iter={k}",flush=_B);l={_K:_V,_N:B,_F:X,_Q:_B,_J:round(R,4)if R is not _A else _A};return round(W,4),i,_Y,A0,l
		m=Z if Z is not _A else F;A2=_m if Z is not _A else _L;I=_A;S=-np.inf;A3=_A;A4=_A;n=_A
		for(M,D)in enumerate(E,1):
			if G:print(f"[MLP] 候補{M}/{len(E)}: alpha={D.get(Y)}, single_layer={D.get(o,_C)}",flush=_B)
			Q,h,A5,X=U(F,m,D);A6=float(r2_score(m[A].values,A5))
			if A6>S:S=A6;I=Q;A3=h;A4=A5;n=X
			yield
		if I is _A:return _A,[],_A,_A,_A
		if A2==_L:R=S
		else:AH=F[B].fillna(F[B].median()).values;AI=I.predict(AH);R=float(r2_score(F[A].values,_invert_y_transform(AI,J,K)))
		i=_mlp_feature_importance(I,m,B,A,J,K)
		with open(os.path.join(p,_u),_T)as j:u.dump({_A6:I,_I:B,_c:A,_F:n},j)
		k=getattr(I[_Y],A7,'?');print(f"[MLP] R²={S:.4f}  hidden={A3}  iter={k}",flush=_B);l={_K:A2,_N:B,_F:n,_Q:_B,_J:round(R,4)};return round(S,4),i,_Y,A4,l
	except Exception as AJ:print(f"[MLP] 失敗: {AJ}",flush=_B);return _A,[],_A,_A,_A
def _try_mlp(*A,**B):
	'同期版の薄いラッパー(exe版が使う)。';C=_try_mlp_steps(*A,**B)
	try:
		while _B:next(C)
	except StopIteration as D:return D.value
async def _try_mlp_async(*A,**B):
	'非同期版の薄いラッパー(Pyodide向け)。';C=_try_mlp_steps(*A,**B)
	try:
		while _B:next(C);await _maybe_yield()
	except StopIteration as D:return D.value
_LGBM_BAG_PARAMS={_e:dict(boosting_type=_e,num_leaves=63,n_estimators=300,bagging_fraction=.7,bagging_freq=1,feature_fraction=.7,min_child_samples=10),_n:dict(boosting_type=_e,num_leaves=63,n_estimators=300,bagging_fraction=.8,bagging_freq=1,feature_fraction=.6,min_child_samples=5,extra_trees=_B)}
_LGBM_BAG_TAG={_e:'LGBM-RF',_n:'LGBM-XT'}
def _try_sktree(kind,df_train,target_col,model_dir,y_transform=_M,y_params={},df_all=_A,splits=_A,num_jobs=4,df_all_per_fold=_A):
	O=splits;N=df_all;M=model_dir;F=y_params;E=y_transform;D=target_col;B=kind
	try:
		import pickle;X=_LGBM_BAG_TAG.get(B,B);Y=f"{B}_model.txt";G=dict(objective=_z,metric='rmse',verbosity=-1,n_jobs=num_jobs,force_col_wise=_B,deterministic=_B,random_state=42);G.update(_LGBM_BAG_PARAMS[B]);A=_get_feat_cols(df_train,D)
		if not A or N is _A or O is _A:return _A,[],_A,_A,_A
		C=N;H=np.zeros(len(C))
		for(Z,(P,Q))in enumerate(O):R,I=_fold_frame(C,df_all_per_fold,Z,P,D,A);J=R.iloc[P];a=R.iloc[Q];S=J[I].median();b=_lgb_fit(G,J[I].fillna(S).values,_apply_y_transform(J[D].values,E,F));c=b.predict(a[I].fillna(S).values);H[Q]=_invert_y_transform(c,E,F)
		T=float(r2_score(C[D].values,H));print(f"[{X}] OOF R²={T:.4f}",flush=_B);K=C[A].median();U=C[A].fillna(K).values;L=_lgb_fit(G,U,_apply_y_transform(C[D].values,E,F));d=L.predict(U);e=float(r2_score(C[D].values,_invert_y_transform(d,E,F)));V={A:float(K[A])if not np.isnan(float(K[A]))else _D for A in A};L.save_model(os.path.join(M,Y))
		with open(os.path.join(M,f"{B}_meta.json"),'w',encoding=_R)as f:json.dump({_I:A,_F:V},f,ensure_ascii=_C)
		W=_lgb_importance(L,len(A));g=max(W.sum(),_E);h=sorted([{_G:A[B],_H:round(float(W[B]/g*100),1)}for B in range(len(A))],key=lambda x:x[_H],reverse=_B);i={_K:_V,_N:A,_F:V,_Q:_B,_J:round(e,4)};return round(T,4),h,B,H,i
	except Exception as j:print(f"[{_LGBM_BAG_TAG.get(B,B)}] 失敗: {j}",flush=_B);return _A,[],_A,_A,_A
def _fit_blend_oof(candidates,y_full):
	'全行OOF予測を持つ候補で NNLS ブレンドを構成する。\n    Returns: (blend_r2, names, weights, blend_oof_preds, feat_list) or None';K=candidates;F=y_full;S=len(F);G={}
	for(L,(N,Z,a,C,O))in K.items():
		if C is _A or N is _A or N<BLEND_R2_THRESH:continue
		if O is _A or O.get(_K)!=_V:continue
		C=np.asarray(C,dtype=float)
		if len(C)!=S or not np.isfinite(C).all():continue
		G[L]=C
	if len(G)<2:return
	D=list(G.keys());B=np.column_stack([G[A]for A in D])
	try:
		from _light import nnls;A,b=nnls(B,F)
		if A.sum()<=1e-09:raise ValueError('NNLS returned all-zero weights')
	except Exception:A=np.clip(np.array([K[A][0]for A in D]),1e-06,_A);A=A/A.sum()
	E=float(r2_score(F,B@A))
	if len(D)>=2:
		try:
			H=B.T@B;T=float(np.mean(np.diag(H)))if H.shape[0]>0 else _E;U=max(1e-06,.05*T);P=np.linalg.solve(H+U*np.eye(H.shape[0]),B.T@F);I=float(r2_score(F,B@P));print(f"[Blend] 符号付きridge候補: OOF R²={I:.4f} (NNLS={E:.4f})",flush=_B)
			if I>E+STACKER_MARGIN:print(f"[Blend] 符号付きridgeを採用(差 {I-E:+.4f} > {STACKER_MARGIN})",flush=_B);A,E=P,I
		except Exception as V:print(f"[Blend] 符号付きridge候補の計算に失敗、NNLSを維持: {V}",flush=_B)
	W=B@A;X=', '.join(f"{A}={B:.3f}"for(A,B)in zip(D,A));print(f"[Blend] members=[{X}] OOF R²={E:.4f}",flush=_B);J={}
	for(L,Q)in zip(D,A):
		if Q<=0:continue
		for M in K[L][1]:J[M[_G]]=J.get(M[_G],_D)+Q*M[_H]
	R=sum(J.values());Y=sorted([{_G:A,_H:round(B/R*100,1)}for(A,B)in J.items()],key=lambda x:x[_H],reverse=_B)if R>0 else[];return E,D,A,W,Y
_F32_ABS_MAX=3.4028235e38
def _clamp_f32(v):
	"実バグ2026-08: |v|がfloat32表現上限を超えると struct.pack('<f', v) が\n    OverflowErrorを送出し、その1個のスカラのために.tregエクスポート全体\n    (=デプロイ候補が全滅)が失敗していた。表現できない値は上限へクランプして\n    書き出し自体は継続する(この規模の値は元々外れ値として学習側で扱われるべき\n    もので、クランプが実用上の予測精度を損なうことはない)。";v=float(v)
	if not math.isfinite(v):return _D
	if v>_F32_ABS_MAX:return _F32_ABS_MAX
	if v<-_F32_ABS_MAX:return-_F32_ABS_MAX
	return v
def _pack_f32(v):return struct.pack('<f',_clamp_f32(v))
def _pack_f32x2(a,b):return struct.pack('<ff',_clamp_f32(a),_clamp_f32(b))
def _write_str_treg(f,s):A=s.encode(_R);f.write(len(A).to_bytes(2,'little'));f.write(A)
def _parse_lgbm_to_treg_bytes(model_txt_path):
	'LightGBM v4 テキスト形式 (`Tree=N` ブロック) をパースして .treg のツリーバイト列を返す。\n    不整合・非対応形式は ValueError を送出する（黙って壊れた出力を返さない）。';R='right_child';Q='left_child';P='threshold';O='split_feature';L='n_leaves';H='leaf_value';import re as I,struct as J
	with open(model_txt_path,encoding=_R)as V:M=V.read()
	U=I.search('^end of trees\\s*$',M,I.MULTILINE);W=M[:U.start()]if U else M;X=bool(I.search('^average_output\\s*$',M,I.MULTILINE));S=I.split('(?m)^Tree=(\\d+)\\s*$',W)
	if len(S)<3:raise ValueError("no 'Tree=N' blocks found (unexpected LightGBM model text format)")
	E=[]
	for A in range(2,len(S),2):
		G={}
		for K in S[A].splitlines():
			K=K.strip()
			if not K or'='not in K:continue
			Y,Z=K.split('=',1);G[Y]=Z
		if int(G.get('num_cat','0')or 0)>0:raise ValueError('categorical split (num_cat>0) is not supported by .treg export')
		F=int(G[_A4])
		if F<1:raise ValueError(f"invalid num_leaves={F}")
		N=[float(A)for A in G.get(H,'').split()]
		if len(N)!=F:raise ValueError(f"leaf_value length {len(N)} != num_leaves {F}")
		if F==1:E.append({L:1,O:[],P:[],Q:[],R:[],H:N});continue
		C=F-1
		def T(key):
			A=[int(A)for A in G.get(key,'').split()]
			if len(A)!=C:raise ValueError(f"{key} length {len(A)} != {C}")
			return A
		def a(key):
			A=[float(A)for A in G.get(key,'').split()]
			if len(A)!=C:raise ValueError(f"{key} length {len(A)} != {C}")
			return A
		E.append({L:F,O:T(O),P:a(P),Q:T(Q),R:T(R),H:N})
	if not E:raise ValueError('parsed 0 trees')
	if X:
		b=_E/len(E)
		for B in E:B[H]=[A*b for A in B[H]]
	D=bytearray();D+=J.pack('<II',len(E),0)
	for B in E:
		C=B[L]-1;D+=J.pack(_O,B[L])
		for A in range(C):D+=J.pack(_O,B[O][A])
		for A in range(C):D+=_pack_f32(B[P][A])
		for A in range(C):D+=J.pack('<i',B[Q][A])
		for A in range(C):D+=J.pack('<i',B[R][A])
		for A in range(B[L]):D+=_pack_f32(B[H][A])
	return bytes(D)
def _load_export_source(model_type,model_dir):
	"モデル種別に応じて (feat_cols, medians, payload, export_type) を pkl / sidecar から\n    自己取得する。export_type は .treg 上の実書式ファミリ:\n      - poly-Ridge は 'linear_poly'（標準化後に多項式展開する専用フォーマット）\n      - LGBM-RF/LGBM-XT は木構造・予測規則が LightGBM ネイティブ形式と同一のため\n        'lgbm' にエイリアスする（LightGBM は boosting_type='rf' でもテキストモデル形式・\n        「木の出力を足し合わせる」推論規則は通常の GBDT と変わらないため）\n    未対応の種別（未知の model_type）のみ None を返す。";C=model_dir;B=model_type;import pickle as F
	if B==_a:
		with open(os.path.join(C,_j),_o)as D:A=F.load(D)
		H=_A7 if A.get(_q)else _a;return A[_I],A.get(_F,{}),A,H
	if B==_X:
		with open(os.path.join(C,_A0),encoding=_R)as G:E=json.load(G)
		return E[_I],E.get(_F,{}),{_w:_s},_X
	if B in(_e,_n):
		with open(os.path.join(C,f"{B}_meta.json"),encoding=_R)as G:E=json.load(G)
		return E[_I],E.get(_F,{}),{_w:f"{B}_model.txt"},_X
	if B==_b:
		with open(os.path.join(C,_t),_o)as D:A=F.load(D)
		return A[_I],A.get(_F,{}),A,_b
	if B==_Y:
		with open(os.path.join(C,_u),_o)as D:A=F.load(D)
		return A[_I],A.get(_F,{}),A,_Y
_TREG_TYPE_MAP={_a:0,_X:1,_b:2,_Y:3,_A7:4,_d:5}
_TREG_OP_MAP={'mul':0,'sq':1,'sign':2}
_CAT_METHOD_CODE={'onehot':0,_A3:1,_A1:2,_A8:3}
_CAT_DATETIME_PART_ID={B:A for(A,B)in enumerate(_DATETIME_PART_NAMES)}
def _write_treg_stream(f,export_type,feat_cols,medians,payload,model_dir,target_col,y_transform,y_params,smear,y_clip,round_output,x_clip_all,derived_recipe,cat_encoders_all=_A):
	"1モデル分の完全な .treg バイト列（'TREG' ヘッダ込み）をファイルライクな f に書く。\n    export_type=='blend' の場合、payload['members'] の各要素を「後処理なしの自己完結した\n    入れ子 .treg ブロブ」として再帰的に埋め込む（この関数自身を再帰呼び出しする）。\n    派生特徴（自動FE）を使うモデルは v4以上（レシピブロック付き）、カテゴリエンコーダを\n    使うモデルは v5（精度レバー4/.treg v5: cat_encodersブロック追加）、datetime_parts\n    エンコーダを使うモデルは v6（真因④対策/.treg v6: cat_encoders method=2追加）、\n    それ以外は v3 で書く。";e=derived_recipe;d=y_clip;c=y_params;b=y_transform;a=target_col;Z=model_dir;O=cat_encoders_all;N=x_clip_all;M='<B';I=feat_cols;G=payload;B=export_type;C=len(I);O=O or[]
	def P(scale):return np.maximum(np.asarray(scale,dtype=np.float64),1e-08).astype(np.float32)
	g=set(I);R=[A for A in e if A[_G]in g and A[_h]in _TREG_OP_MAP];J=[A for A in O if A[_S]in g];r=any(A.get(_f)==_A8 for A in J);s=any(A.get(_f)==_A1 for A in J);S=7 if r else 6 if s else 5 if J else 4 if R else 3;f.write(b'TREG');f.write(struct.pack('<BB',S,_TREG_TYPE_MAP[B]));f.write(struct.pack(_O,C))
	if S>=4:
		f.write(struct.pack(_O,len(R)))
		for T in R:U=T[_i];h=U[0];V=U[1]if len(U)>1 else'';t,u=N.get(h,(-X_CLIP_SENTINEL,X_CLIP_SENTINEL));v,w=N.get(V,(-X_CLIP_SENTINEL,X_CLIP_SENTINEL))if V else(-X_CLIP_SENTINEL,X_CLIP_SENTINEL);f.write(struct.pack(M,_TREG_OP_MAP[T[_h]]));_write_str_treg(f,T[_G]);_write_str_treg(f,h);f.write(_pack_f32x2(t,u));_write_str_treg(f,V);f.write(_pack_f32x2(v,w))
	if S>=5:
		f.write(struct.pack(_O,len(J)))
		for E in J:
			W=_CAT_METHOD_CODE[E[_f]];f.write(struct.pack(M,W));_write_str_treg(f,E[_S]);_write_str_treg(f,E[_P])
			if W==0:_write_str_treg(f,E[_AF])
			elif W in(1,3):
				i=E['map'];f.write(struct.pack(_O,len(i)))
				for(x,y)in i.items():_write_str_treg(f,x);f.write(_pack_f32(y))
				f.write(_pack_f32(E[_A2]))
			else:f.write(struct.pack(M,_CAT_DATETIME_PART_ID[E['part']]))
	if B==_a:A=G;f.write(np.array(A[_U].center_,dtype=np.float32).tobytes());f.write(P(A[_U].scale_).tobytes());f.write(np.array(A[_Z].coef_,dtype=np.float32).tobytes());f.write(_pack_f32(A[_Z].intercept_))
	elif B==_A7:
		A=G;D=A[_U];j=A[_Z];f.write(np.array(D.center_,dtype=np.float32).tobytes());f.write(P(D.scale_).tobytes());X=[(A,-1)for A in range(C)]
		for Q in range(C):
			for z in range(Q,C):X.append((Q,z))
		f.write(struct.pack(_O,len(X)))
		for(A0,A1)in X:f.write(struct.pack('<ii',A0,A1))
		f.write(np.array(j.coef_,dtype=np.float32).tobytes());f.write(_pack_f32(j.intercept_))
	elif B==_X:A2=_parse_lgbm_to_treg_bytes(os.path.join(Z,G[_w]));f.write(A2)
	elif B==_b:
		A=G;D=A[_U];F=A[_Z];A3=float(F.sigma_var);K=np.atleast_1d(np.array(F.length_scale,dtype=float))
		if len(K)!=C:K=np.full(C,float(K.mean())if len(K)else _E)
		f.write(np.array(D.mean_,dtype=np.float32).tobytes());f.write(P(D.scale_).tobytes());f.write(K.astype(np.float32).tobytes());f.write(_pack_f32(A3));A4=float(getattr(F,'y_mean_',_D));A5=float(getattr(F,'y_std_',_E));f.write(_pack_f32x2(A4,A5));A6=len(F.X_train_);f.write(struct.pack(_O,A6));f.write(F.X_train_.astype(np.float32).tobytes());f.write(F.alpha_.astype(np.float32).tobytes())
	elif B==_Y:
		A=G;k=A[_A6];D=k[_U];Y=k[_Y];f.write(np.array(D.mean_,dtype=np.float32).tobytes());f.write(P(D.scale_).tobytes());l=len(Y.coefs_);f.write(struct.pack(_O,l))
		for(Q,(m,A7))in enumerate(zip(Y.coefs_,Y.intercepts_)):A8,A9=m.shape;AA=1 if Q==l-1 else 0;f.write(struct.pack('<IIB',A8,A9,AA));f.write(m.astype(np.float32).tobytes());f.write(A7.astype(np.float32).tobytes())
	elif B==_d:
		import io;n=G[_A9];f.write(struct.pack(_O,len(n)))
		for L in n:o=io.BytesIO();_write_treg_stream(o,L[_AA],L[_I],L[_F],L[_AB],Z,a,b,c,_E,(-X_CLIP_SENTINEL,X_CLIP_SENTINEL),_C,N,e,O);p=o.getvalue();f.write(_pack_f32(L[_x]));f.write(struct.pack(_O,len(p)));f.write(p)
	else:raise ValueError(f"unknown export_type: {B}")
	AB={_M:0,_k:1,_l:2};q=_M if B==_d else b;f.write(struct.pack(M,AB.get(q,0)))
	if q==_l:f.write(_pack_f32(c.get(_p,_E)))
	f.write(struct.pack(M,1 if round_output else 0));f.write(_pack_f32(smear));f.write(_pack_f32x2(d[0],d[1]));f.write(struct.pack(_O,C))
	for H in I:AC,AD=N.get(H,(-X_CLIP_SENTINEL,X_CLIP_SENTINEL));f.write(_pack_f32x2(AC,AD))
	_write_str_treg(f,a);f.write(struct.pack(_O,C))
	for H in I:_write_str_treg(f,H)
	f.write(struct.pack(_O,C))
	for H in I:_write_str_treg(f,H);f.write(struct.pack('<d',float(medians.get(H,_D))))
def _export_treg(model_type,model_dir,target_col,y_transform=_M,y_params=_A,smear=_E,y_clip=(-X_CLIP_SENTINEL,X_CLIP_SENTINEL),round_output=_C,x_clip_all=_A,derived_recipe=_A,cat_encoders_all=_A):
	'モデル pkl / sidecar から実使用列・median を自己取得し、次元整合した .treg を書く。\n    linear(poly含む)/lgbm/gp/mlp/rf/xt 全種別に対応(blend は _export_treg_blend を使う)。';G=cat_encoders_all;F=derived_recipe;E=x_clip_all;D=y_params;C=model_dir;A=model_type;D=D or{};E=E or{};F=F or[];G=G or[];H=os.path.join(C,_AC);B=H+'.tmp'
	try:
		I=_load_export_source(A,C)
		if I is _A:print(f"[TREG] {A} は非対応のためスキップ",flush=_B);return _C
		J,K,L,M=I
		with open(B,_T)as N:_write_treg_stream(N,M,J,K,L,C,target_col,y_transform,D,smear,y_clip,round_output,E,F,G)
		os.replace(B,H);O=os.path.getsize(H)//1024;print(f"[TREG] {A} → model.treg ({O} KB, {len(J)}特徴量)",flush=_B);return _B
	except Exception as P:
		print(f"[TREG] エクスポート失敗 ({A}): {P}",flush=_B)
		try:
			if os.path.exists(B):os.remove(B)
		except Exception:pass
		return _C
def _export_treg_blend(model_dir,target_col,candidates,y_transform=_M,y_params=_A,smear=_E,y_clip=(-X_CLIP_SENTINEL,X_CLIP_SENTINEL),round_output=_C,x_clip_all=_A,derived_recipe=_A,cat_encoders_all=_A):
	'blend_meta.pkl のメンバー構成をもとに、各メンバーを自己完結した入れ子 .treg として\n    埋め込んだアンサンブル用 .treg を書く。1メンバーでも書き出せなければ全体を失敗とする\n    （部分的なブレンドは学習時に最適化した重み構成と乖離するため中途半端な出力を避ける）。';K=candidates;H=cat_encoders_all;G=derived_recipe;F=x_clip_all;E=y_params;B=model_dir;import pickle as O;E=E or{};F=F or{};G=G or[];H=H or[];I=os.path.join(B,_AC);C=I+'.tmp'
	try:
		with open(os.path.join(B,_AD),_o)as J:L=O.load(J)
		P=L['models'];Q=L.get('weights',{});D=[]
		for A in P:
			if A not in K:raise ValueError(f"Blend サブモデル '{A}' が候補に見つかりません")
			M=K[A][2];N=_load_export_source(M,B)
			if N is _A:raise ValueError(f"Blend サブモデル '{A}' ({M}) を書き出せません")
			R,S,T,U=N;D.append({_AA:U,_I:R,_F:S,_AB:T,_x:float(Q.get(A,_D))})
		if len(D)<2:raise ValueError('Blend の書き出し可能サブモデルが2未満です')
		with open(C,_T)as J:_write_treg_stream(J,_d,[],{},{_A9:D},B,target_col,y_transform,E,smear,y_clip,round_output,F,G,H)
		os.replace(C,I);V=os.path.getsize(I)//1024;print(f"[TREG] blend({len(D)}メンバー) → model.treg ({V} KB)",flush=_B);return _B
	except Exception as W:
		print(f"[TREG] エクスポート失敗 (blend): {W}",flush=_B)
		try:
			if os.path.exists(C):os.remove(C)
		except Exception:pass
		return _C
def _export_treg_lgbm_bag(model_dir,target_col,feat_cols,medians,n_folds,y_transform=_M,y_params=_A,smear=_E,y_clip=(-X_CLIP_SENTINEL,X_CLIP_SENTINEL),round_output=_C,x_clip_all=_A,derived_recipe=_A,cat_encoders_all=_A):
	'対策(2026-07 第2弾・真因①): LightGBM の fold バギングを .treg に書き出す。\n    K本のfoldモデル(_try_lgbm_steps が既にOOF評価のために学習済み、追加学習は不要)を\n    等重み(1/K)の blend として書く。.treg のblendリーダー(native C++/JS)は元々\n    「N個の自己完結した入れ子モデルの加重和」という汎用フォーマットで、メンバーの\n    由来がヘテロなブレンドかfoldバギングかを区別しないため、エンジン側の変更は一切\n    不要(writer側のみで完結する)。';G=cat_encoders_all;F=derived_recipe;E=x_clip_all;D=y_params;C=n_folds;B=model_dir;D=D or{};E=E or{};F=F or[];G=G or[];H=os.path.join(B,_AC);A=H+'.tmp'
	try:
		I=[]
		for K in range(C):
			J=f"lgbm_model_fold{K}.txt"
			if not os.path.exists(os.path.join(B,J)):raise ValueError(f"バギングメンバー '{J}' が見つかりません")
			I.append({_AA:_X,_I:feat_cols,_F:medians,_AB:{_w:J},_x:_E/C})
		if len(I)<2:raise ValueError('バギングメンバーが2未満です')
		with open(A,_T)as L:_write_treg_stream(L,_d,[],{},{_A9:I},B,target_col,y_transform,D,smear,y_clip,round_output,E,F,G)
		os.replace(A,H);M=os.path.getsize(H)//1024;print(f"[TREG] lgbm_bag({C}fold) → model.treg ({M} KB)",flush=_B);return _B
	except Exception as N:
		print(f"[TREG] エクスポート失敗 (lgbm_bag): {N}",flush=_B)
		try:
			if os.path.exists(A):os.remove(A)
		except Exception:pass
		return _C
SIM_MAX_LEVELS=30
SIM_TRACK_BINS=28
SIM_MAX_YHIST_BINS=40
SIM_MIN_YHIST_BINS=10
SIM_NEIGHBOR_ROWS=300
SIM_NEIGHBOR_K=10
SIM_NEIGHBOR_PROBE=120
SIM_CORR_MIN=.7
SIM_MAX_CORR_PAIRS=20
def _sim_round_sig1(v):
	'スライダー step 用に有効数字1桁へ丸める。0や非有限は安全な既定値に落とす。';v=float(v)
	if not np.isfinite(v)or v<=0:return _E
	A=math.floor(math.log10(v));return float(round(v/10**A)*10**A)or float(10**A)
def _sim_round_sig(v,sig=12):
	'実バグ2026-08: SIMULATE用JSONの丸めが固定小数桁(round(x,6)等)だと、\n    値の絶対値が1e-6未満のスケールの列(nm単位の計測値等)でmin/max/std/傾きが\n    全て0.0に潰れる。固定小数桁でなく有効数字sig桁で丸め、スケールに依存せず\n    情報を保持する。0/非有限はそのまま返す。\n    sig=12(旧: 6): sig=6では典型的なターゲットスケール(例: 値~20台)でも小数点\n    以下4桁程度しか残らず絶対誤差が1e-5に達し、seed_rows.yを元CSVの実データと\n    突き合わせて同一行を特定する既存の許容誤差(1e-6)を壊してしまう\n    (tests/test_simulate_spec.py で実測検出)。sig=12なら典型スケールで\n    絶対誤差~1e-11に収まりつつ、超微小スケールでも0への丸め崩壊を防げる。';v=float(v)
	if not np.isfinite(v)or v==0:return v
	A=math.floor(math.log10(abs(v)));B=sig-1-A;return float(round(v,B))
def _sim_aggregate_importance(feat_list_full,raw_cols,cat_onehot_specs,cat_datetime_specs,numkey_col_name,numkey_source_cols,derived_recipe):
	'engineered な特徴量重要度を「生CSV列」へ集約する。\n\n    feature_importance の name は one-hot(`col=value`)・target encoding・\n    datetime_parts・派生(`a*b` / `a^2` / `sign(a)`)・poly(`a b` / `a^2`)が混在しており、\n    そのままではスライダー(=生列)の並び順にも近傍距離の重みにも使えない。\n    由来をたどって生列へ配分し、合計100%に正規化して返す。\n    集約は「上位10件に切る前の全量」に対して行う(切ってから集約すると寄与が欠ける)。\n    ';F=numkey_col_name;L=set(raw_cols);B={}
	for A in cat_onehot_specs or[]:B[A[_S]]=[A[_P]]
	for A in cat_datetime_specs or[]:B[A[_S]]=[A[_P]]
	for G in derived_recipe or[]:B[G[_G]]=list(G.get(_i)or[])
	if F:B[F]=list(numkey_source_cols or[])
	def D(name,depth=0):
		E=depth;A=name
		if E>4 or not A:return[]
		if A in L:return[A]
		if A in B:
			C=[]
			for G in B[A]:C.extend(D(G,E+1))
			return C
		if A.endswith('^2'):return D(A[:-2],E+1)
		if _g in A:
			F=[D(A,E+1)for A in A.split(_g)if A]
			if F and all(F):
				C=[]
				for H in F:C.extend(H)
				return C
		return[]
	C={}
	for H in feat_list_full or[]:
		E=D(str(H.get(_G,'')))
		if not E:continue
		I=float(H.get(_H)or _D)
		if I<=0:continue
		M=I/len(E)
		for J in E:C[J]=C.get(J,_D)+M
	K=sum(C.values())
	if K<=0:return{}
	return{A:B/K*1e2 for(A,B)in C.items()}
def _sim_build_slider_spec(df_raw,used_raw_cols,raw_imp,x_clip_bounds):
	'生CSV列1本 = スライダー1本の仕様を作る(重要度降順)。';O='value';K=df_raw;J='importance_pct';H=x_clip_bounds;F=[];H=H or{}
	for C in used_raw_cols:
		if C not in K.columns:continue
		E=K[C];L=round(float(raw_imp.get(C,_D)),2);P=pd.api.types.is_numeric_dtype(E)and not pd.api.types.is_bool_dtype(E)
		if P:
			A=pd.to_numeric(E,errors=_W).values.astype(float);A=A[np.isfinite(A)]
			if A.size==0:continue
			B,D=float(A.min()),float(A.max())
			if not D>B:D=B+max(abs(B)*1e-06,1e-06)
			Q,R,S=(float(A)for A in np.percentile(A,[1,50,99]));T,W=np.histogram(A,bins=SIM_TRACK_BINS,range=(B,D));M=bool(np.all(np.mod(A,_E)==_D));U=_E if M else _sim_round_sig1((D-B)/3e2);G=H.get(C);F.append({_y:C,'kind':'numeric',J:L,'median':_sim_round_sig(R),'p1':_sim_round_sig(Q),'p99':_sim_round_sig(S),'min':_sim_round_sig(B),'max':_sim_round_sig(D),'step':U,'hist':[int(A)for A in T],'hist_lo':_sim_round_sig(B),'hist_hi':_sim_round_sig(D),'x_clip_lo':_sim_round_sig(float(G[0]))if G else _A,'x_clip_hi':_sim_round_sig(float(G[1]))if G else _A,'is_integer':M})
		else:
			V=E.where(E.notna(),CAT_NAN_SENTINEL).astype(str);N=V.value_counts()
			if N.empty:continue
			I=[{O:str(A),'count':int(B)}for(A,B)in N.items()];F.append({_y:C,'kind':'categorical',J:L,'mode':I[0][O],'levels':I[:SIM_MAX_LEVELS],'truncated':bool(len(I)>SIM_MAX_LEVELS)})
	F.sort(key=lambda d:d[J],reverse=_B);return F
def _sim_build_y_hist(y_all):
	'ターゲットの分布。scatter.true は250点サブサンプル+評価行のみのため使えない。';A=np.asarray(y_all,dtype=float);A=A[np.isfinite(A)]
	if A.size==0:return
	B,C=float(A.min()),float(A.max())
	if not C>B:C=B+max(abs(B)*1e-06,1e-06)
	F,G=(float(A)for A in np.percentile(A,[25,75]));D=2.*(G-F)/A.size**(_E/3.)if A.size>0 else _D
	if D>0:E=int(min(SIM_MAX_YHIST_BINS,max(SIM_MIN_YHIST_BINS,math.ceil((C-B)/D))))
	else:E=20
	H,I=np.histogram(A,bins=E,range=(B,C));J,K,L=(float(A)for A in np.percentile(A,[10,50,90]));return{'bin_edges':[_sim_round_sig(float(A))for A in I],'counts':[int(A)for A in H],'n':int(A.size),'p10':_sim_round_sig(J),'p50':_sim_round_sig(K),'p90':_sim_round_sig(L)}
def _sim_build_seed_rows(df_raw,target_col,spec_cols):
	'スライダーの出発点にする「実在する行」。全列中央値の合成行は、相関のある\n    データでは実在しない組み合わせになりうるため使わない。';G=target_col;A=df_raw
	if G not in A.columns or len(A)==0:return[]
	C=pd.to_numeric(A[G],errors=_W).values.astype(float);D=np.where(np.isfinite(C))[0]
	if D.size==0:return[]
	B=D[np.argsort(C[D],kind='stable')]
	def E(q):return int(B[int(round((B.size-1)*q))])
	L=np.random.RandomState(42);M=[('low',E(.1)),('median',E(.5)),('high',E(.9)),('random',int(B[L.randint(B.size)]))];H=[]
	for(N,I)in M:
		O=A.iloc[I];J={}
		for F in spec_cols:
			if F not in A.columns:continue
			K=O[F];J[F]=CAT_NAN_SENTINEL if pd.isna(K)else str(K)
		H.append({'label':N,'y':_sim_round_sig(float(C[I])),'values':J})
	return H
def _sim_build_neighbor_ref(df_raw,numeric_cols,raw_imp):
	'近傍件数ゲージ用の参照データ。\n\n    重要度による次元の重み付けが必須(モック実測): 素朴なユークリッド距離だと\n    数値列が数十本あるデータで「実在する学習行ですら近傍0件」になる(次元の呪い。\n    無関係な列のノイズが距離を支配する)。\n    半径は「各行からk番目に近い行までの重み付き距離」の中位数とする。この定義なら\n    実在行は概ねk件前後の近傍を持ち、非現実的な組み合わせは0件になり、かつ\n    次元数に依存しない。\n    ';G=df_raw;C=[A for A in numeric_cols if A in G.columns]
	if not C or len(G)<2:return
	B=G[C].apply(pd.to_numeric,errors=_W).values.astype(float);L=np.isfinite(B).all(axis=1);B=B[L]
	if B.shape[0]<2:return
	H=B.mean(axis=0);D=B.std(axis=0);D[D<1e-12]=_E;A=(B-H)/D
	if A.shape[0]>SIM_NEIGHBOR_ROWS:M=int(math.ceil(A.shape[0]/SIM_NEIGHBOR_ROWS));A=A[::M]
	I=np.array([max(float(raw_imp.get(A,_D)),.01)for A in C],dtype=float);J=I/I.sum()*len(C);N=A[:min(SIM_NEIGHBOR_PROBE,A.shape[0])];O=((N[:,_A,:]-A[_A,:,:])**2*J).sum(axis=2);E=np.sqrt(np.maximum(O,_D));E.sort(axis=1);K=min(SIM_NEIGHBOR_K,E.shape[1]-1);F=float(np.median(E[:,K]))if K>=1 else float(np.median(E[:,-1]))
	if not np.isfinite(F)or F<=0:F=_E
	return{_i:C,'mean':[_sim_round_sig(float(A))for A in H],'std':[_sim_round_sig(float(A))for A in D],_x:[round(float(A),4)for A in J],'rows':[[round(float(A),4)for A in A]for A in A],'radius':_sim_round_sig(F),'k':SIM_NEIGHBOR_K}
def _sim_build_corr_pairs(df_raw,numeric_cols):
	'強相関ペア。連動モードとゴースト表示(相関相手が示す「あるべき値」)に使う。';N=df_raw;D=[A for A in numeric_cols if A in N.columns]
	if len(D)<2:return[]
	A=N[D].apply(pd.to_numeric,errors=_W).values.astype(float);Q=np.isfinite(A).all(axis=1);A=A[Q]
	if A.shape[0]<3:return[]
	E=A.mean(axis=0);G=A-E;H=(G**2).mean(axis=0);F=np.sqrt(np.maximum(H,_D));I=[]
	for B in range(len(D)):
		if F[B]<1e-12:continue
		for C in range(B+1,len(D)):
			if F[C]<1e-12:continue
			J=float((G[:,B]*G[:,C]).mean());K=J/(F[B]*F[C])
			if not np.isfinite(K)or abs(K)<SIM_CORR_MIN:continue
			L=J/max(H[B],1e-12);O=float(E[C]-L*E[B]);M=J/max(H[C],1e-12);P=float(E[B]-M*E[C]);R=float(np.sqrt(np.mean((A[:,C]-(L*A[:,B]+O))**2)));S=float(np.sqrt(np.mean((A[:,B]-(M*A[:,C]+P))**2)));I.append({'a':D[B],'b':D[C],'r':round(float(K),3),'sAB':_sim_round_sig(float(L)),'iAB':_sim_round_sig(O),'sBA':_sim_round_sig(float(M)),'iBA':_sim_round_sig(P),'sdA':_sim_round_sig(S),'sdB':_sim_round_sig(R)})
	I.sort(key=lambda d:abs(d['r']),reverse=_B);return I[:SIM_MAX_CORR_PAIRS]
def _sim_used_raw_columns(df_raw,target_col,feat_cols_all,cat_onehot_specs,cat_target_cols,cat_datetime_specs,cat_dropped_cols,numkey_col_name,numkey_source_cols):
	'モデルが実際に使っている生CSV列を洗い出す。\n    除外: ターゲット / 高カーディナリティで捨てた列 / 定数・重複で落ちた列 /\n          合成キーのような内部生成列。';F=numkey_col_name;D=df_raw;E=set(feat_cols_all or[]);A=set()
	for B in E:
		if B in D.columns:A.add(B)
	for C in cat_onehot_specs or[]:
		if C[_S]in E:A.add(C[_P])
	for C in cat_datetime_specs or[]:
		if C[_S]in E:A.add(C[_P])
	for B in cat_target_cols or[]:
		if B==F:
			for G in numkey_source_cols or[]:A.add(G)
		elif B in D.columns:A.add(B)
	A.discard(target_col)
	for B in cat_dropped_cols or[]:A.discard(B)
	A.discard(F);return[B for B in D.columns if B in A]
async def _run_main():
	'学習処理の本体。exe版はCPython上でThreadPoolExecutorによる真の並列で実行し、\n    Pyodide(Web版)は_IS_PYODIDE分岐でモデル毎の非同期版(_try_xxx_async)を順番にawaitすることで、\n    候補/fold単位でブラウザに制御を返す(offline.htmlのロボアニメーション改善)。\n    どちらの経路でも計算ロジック自体(_try_xxx_steps)は共通で、結果は変わらない。';Ba='r2_std';BZ='is_best';BY='n_rows';BX='model_meta.json';BW='min_rows';BV='instant';Au='Blend (Ensemble)';AQ='r2';AD='model_type';a='params';Z='key';sys.stdout.reconfigure(line_buffering=_B,encoding=_R)
	if len(sys.argv)<2:_error_exit('csv_path_missing')
	Av=sys.argv[1];A=sys.argv[2]if len(sys.argv)>2 else _A;AE=sys.argv[4]if len(sys.argv)>4 else'quick';K=_NUM_JOBS;H=AE=='thorough';print(f"[Python] CSV を解析中... {Av}",flush=_B);_emit_progress(3,'reading_csv');B=_read_csv_with_encoding_fallback(Av);Aw=B.columns[B.columns.duplicated()].unique().tolist()
	if Aw:_error_exit('duplicate_columns',cols=Aw)
	A2=int(B.duplicated().sum())
	if A2>0:print(f"[Python] 重複行を検出: {A2} 件（削除はせず学習に使用します）",flush=_B)
	B,A,AR=_resolve_and_validate_target(B,A)
	if AE==BV and len(B)>INSTANT_MAX_TRAIN_ROWS:Bb=len(B);B=B.sample(n=INSTANT_MAX_TRAIN_ROWS,random_state=INSTANT_SAMPLE_SEED).reset_index(drop=_B);print(f"[Python] 瞬速モード: 学習行数を {Bb} → {INSTANT_MAX_TRAIN_ROWS} にサブサンプル",flush=_B)
	A3=B.copy();B,c,b,d,k=_prepare_categoricals(B,A)
	if c or b or k:print(f"[Python] カテゴリ列検出: one-hot={sorted(set(A[_P]for A in c))} target_enc={b} datetime={sorted(set(A[_P]for A in k))}",flush=_B)
	if d:print(f"[Python] カテゴリ列除外(高カーディナリティ): {d}",flush=_B)
	Bc={A[_S]for A in c}|{A[_S]for A in k};B,l,AS=_detect_numeric_composite_key(B,A,exclude_cols=Bc)
	if l:b=b+[l]
	L=len(B);Bd='じっくり'if H else'瞬速'if AE==BV else'お急ぎ';print(f"[Python] {L} 行 / {B.shape[1]} 列 / モード: {Bd} / CPU並列: {K}",flush=_B);Ax=bool(np.all(np.mod(B[A].dropna().values,_E)==_D));T=B[A].values.copy();Be=_get_feat_cols(B,A);Bf,Bg=_find_constant_and_duplicate_cols(B,Be);AT=Bf+Bg
	if AT:print(f"[Python] 定数/重複列を除去: {AT}",flush=_B);B=B.drop(columns=AT)
	O=H or L<SMALL_N_OOF_THRESH;U=L>=MIN_ROWS_FOR_SPLIT
	if U:Bh=(10 if L<100 else 5)if O else 5;V=_make_binned_splits(B,A,n_splits=Bh);W,m=V[0]
	else:V=_A;W=np.arange(L);m=np.array([],dtype=int)
	AU=B;AF=0;Ay=_C;e=_A
	if len(W)>=20:
		AV=OUTLIER_IQR_MULT if H else OUTLIER_IQR_QUICK;e=AV
		if U and V is not _A:
			try:e=_select_y_winsorize_cv(B,A,V,AV,num_jobs=K)
			except Exception as f:print(f"[YWinsorize-CV] 選択失敗 → 従来どおりクリップ: {f}",flush=_B);e=AV
		if e is not _A:Bi,Bj=_fit_y_winsorize_bounds(B[A].values[W],e);B,AF=_apply_y_winsorize(B,A,Bi,Bj)
		else:Ay=_B
	I=_M;J={}
	try:
		if H and U and V is not _A:I,J=_select_y_transform_cv(B,A,V,num_jobs=K)
		else:I,J=_detect_y_transform(B[A].values[W],B[A].values)
	except Exception as f:print(f"[YTransform] 検出失敗 → 変換スキップ: {f}",flush=_B)
	if U:C=B.iloc[W].reset_index(drop=_B);G=B.iloc[m].reset_index(drop=_B);print(f"[Python] 層化分割: 学習={len(C)} 行 / 検証={len(G)} 行",flush=_B)
	else:C,G=B,_A;print(f"[Python] データが少ないため全データを学習に使用",flush=_B)
	Az=_get_feat_cols(C,A);N=_compute_x_clip(C,Az)
	if N:
		print(f"[Python] X クリッピング ({X_CLIP_PCTILE[0]}%–{X_CLIP_PCTILE[1]}%): {len(N)} 列",flush=_B);C=_apply_x_clip(C,N)
		if G is not _A:G=_apply_x_clip(G,N)
	n=_apply_x_clip(B,N)if N else B;Bz=n;o=_fit_target_encoders(C,A,b)
	if o:
		C=_apply_target_encoders(C,o);n=_apply_target_encoders(n,o)
		if G is not _A:G=_apply_target_encoders(G,o)
	if l:
		for AW in o:
			if AW[_S]==l:AW[_f]=_A8;AW[_P]=NUMERIC_KEY_SEP.join(AS);break
	A4=c+k+o;P=[]
	if H and L>=FE_MIN_ROWS:
		P=_build_derived_recipe(C,A,num_jobs=K)
		if P:
			C=_apply_derived(C,P);n=_apply_derived(n,P)
			if G is not _A:G=_apply_derived(G,P)
	R=n if O and U else _A;X=V if O and U else _A;Q=_A;p=_A
	if O and U:
		_emit_progress(8,'fold_feature_select');Q=[];p=[]
		for(q,B_)in V:
			if e is not _A:Bk,Bl=_fit_y_winsorize_bounds(AU[A].values[q],e);AG,r=_apply_y_winsorize(AU,A,Bk,Bl)
			else:AG=AU
			A_=_compute_x_clip(AG.iloc[q],Az);AX=_apply_x_clip(AG,A_)if A_ else AG;B0=_fit_target_encoders(AX.iloc[q],A,b);AY=_apply_target_encoders(AX,B0)if B0 else AX;Bm=AY.iloc[q];AZ=[]
			if H and len(q)>=FE_MIN_ROWS:AZ=_build_derived_recipe(Bm,A,num_jobs=K)
			B1=_apply_derived(AY,AZ)if AZ else AY;Q.append(B1);p.append(_lgbm_feature_screen(B1.iloc[q],A,num_jobs=K))
	B2=os.path.dirname(os.path.abspath(__file__));Aa=os.path.join(B2,'trained_model');D=os.path.join(B2,'trained_model_tmp')
	if os.path.exists(D):shutil.rmtree(D)
	os.makedirs(D,exist_ok=_B);_emit_progress(12,'preprocess_done');AH=_lgbm_feature_screen(C,A,num_jobs=K);E={}
	def Ab():return _try_linear(C,G,A,D,I,J,df_all=R,use_oof=O,splits=X,df_all_per_fold=Q)
	def B3():return _try_lgbm(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,num_jobs=K,splits=X,prog=(20,55)if H else _A,df_all_per_fold=Q)
	def B4():return _try_gp(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,feat_cols_override=AH,splits=X,df_all_per_fold=Q,screen_cols_per_fold=p)
	def B5():
		if len(C)<MLP_MIN_ROWS:return _A,[],_A,_A,_A
		return _try_mlp(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,feat_cols_override=AH,splits=X,df_all_per_fold=Q,screen_cols_per_fold=p)
	async def B6():return await _try_lgbm_async(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,num_jobs=K,splits=X,prog=(20,55)if H else _A,df_all_per_fold=Q)
	async def B7():return await _try_gp_async(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,feat_cols_override=AH,splits=X,df_all_per_fold=Q,screen_cols_per_fold=p)
	async def B8():
		if len(C)<MLP_MIN_ROWS:return _A,[],_A,_A,_A
		return await _try_mlp_async(C,G,A,D,use_grid=H,use_oof=O,y_transform=I,y_params=J,df_all=R,feat_cols_override=AH,splits=X,df_all_per_fold=Q,screen_cols_per_fold=p)
	if H:
		_emit_progress(15,'linear_fit');A5=Ab();_emit_progress(20,'lgbm_fit');A6=await B6()if _IS_PYODIDE else B3();_emit_progress(58,'mlp_gp_fit')
		if _IS_PYODIDE:s=await B7();t=await B8()
		else:
			with _thread_limit(max(1,K//2)):
				with ThreadPoolExecutor(max_workers=2)as g:Ac=g.submit(B4);Ad=g.submit(B5);s=Ac.result();t=Ad.result()
	else:
		_emit_progress(20,'parallel4_fit')
		if _IS_PYODIDE:A5=Ab();A6=await B6();s=await B7();t=await B8()
		else:
			with _thread_limit(max(1,K//2)):
				with ThreadPoolExecutor(max_workers=4)as g:Bn=g.submit(Ab);Bo=g.submit(B3);Ac=g.submit(B4);Ad=g.submit(B5);A5=Bn.result();A6=Bo.result();s=Ac.result();t=Ad.result()
	if A5[0]is not _A and np.isfinite(A5[0]):E['Linear (Ridge)']=A5
	if A6[0]is not _A and np.isfinite(A6[0]):E['LightGBM']=A6
	if s[0]is not _A and np.isfinite(s[0]):E['GaussianProcess (ARD-RBF)']=s
	A7=_A
	if t[0]is not _A and np.isfinite(t[0]):E['MLP']=t
	elif len(C)<MLP_MIN_ROWS:A7=BW
	else:A7='failed'
	if H and U and R is not _A and os.environ.get('TREG_NO_SKTREE')!='1':
		_emit_progress(78,'extra_models_fit');Ae=_try_sktree(_e,C,A,D,y_transform=I,y_params=J,df_all=R,splits=X,num_jobs=K,df_all_per_fold=Q)
		if Ae[0]is not _A and np.isfinite(Ae[0]):E['LGBM-RF']=Ae
		Af=_try_sktree(_n,C,A,D,y_transform=I,y_params=J,df_all=R,splits=X,num_jobs=K,df_all_per_fold=Q)
		if Af[0]is not _A and np.isfinite(Af[0]):E['LGBM-XT']=Af
	if H and U and len(E)>=2:
		_emit_progress(88,'ensemble_optimize');B9=_fit_blend_oof(E,B[A].values)
		if B9 is not _A:
			BA,AI,BB,Bp,Bq=B9
			if np.isfinite(BA):Ag=[E[A][4].get(_J)for A in AI if E[A][4]is not _A];BC=float(np.average(Ag,weights=BB))if len(Ag)==len(AI)and all(A is not _A for A in Ag)else _A;Br={_K:_V,_N:[],_F:{},_Q:_B,_J:round(BC,4)if BC is not _A else _A};E[Au]=round(BA,4),Bq,_d,Bp,Br
			import pickle as Bs
			with open(os.path.join(D,_AD),_T)as h:Bs.dump({'models':AI,'weights':{A:float(B)for(A,B)in zip(AI,BB)},'normalize':_C,'version':2},h)
	if not E:_error_exit('no_valid_model')
	M=max(E,key=lambda k:E[k][0]if np.isfinite(E[k][0])else-np.inf)
	if M==Au:
		A8={A:B for(A,B)in E.items()if A!=Au}
		if A8:
			Ah=max(A8,key=lambda k:A8[k][0]if np.isfinite(A8[k][0])else-np.inf);BD=E[M][0]-A8[Ah][0]
			if BD<BLEND_MARGIN:print(f"[Blend] 単体最良 ({Ah}) とのOOF差 {BD:+.4f} < {BLEND_MARGIN} → 安定性を優先し単体モデルを採用",flush=_B);M=Ah
	Ai,BE,u,A9,AJ=E[M];print(f"[Python] 最良モデル: {M} (R²={Ai:.4f})",flush=_B);AA=_get_feat_cols(C,A);Bt=C[AA].median().to_dict();Bu={B:float(A)if not np.isnan(float(A))else _D for(B,A)in Bt.items()}
	with open(os.path.join(D,'impute_medians.json'),'w',encoding=_R)as h:json.dump(Bu,h,ensure_ascii=_C)
	Aj=AJ.get(_K,_L)if AJ else _L;AB=_y_true_for(Aj,T,W,m);BF,v,w,x=_fit_postprocess_params(A9,AB,I,T,Ax);AK=Ai;BG=_A
	if A9 is not _A and AB is not _A and len(AB)==len(A9):
		Ak=_apply_postprocess(A9,BF,v,w,x);Al,Am,An,BH,r=_eval_metrics(Ak,Aj,T,W,m);AK=round(float(r2_score(AB,Ak)),4);Y=np.asarray(AB,dtype=float);y=np.asarray(Ak,dtype=float);BI=np.isfinite(Y)&np.isfinite(y);Y,y=Y[BI],y[BI]
		if len(Y)>0:
			BJ=250
			if len(Y)>BJ:BK=np.random.RandomState(42).choice(len(Y),BJ,replace=_C);Y,y=Y[BK],y[BK]
			BG={'true':[round(float(A),4)for A in Y],'pred':[round(float(A),4)for A in y]}
	else:Al,Am,An,BH,r=_eval_metrics(A9,Aj,T,W,m)
	Ao=_sanitize_json({AD:u,_I:AA,'model_feat_cols':AJ.get(_N,AA)if AJ else AA,_c:A,AQ:AK,'model_label':M,'y_transform':I,'y_params':J,'cat_encoders':A4,'x_clip':N,'derived_features':P,'postprocess':{'smear':BF,'y_clip':[v,w],'round_output':x}})
	with open(os.path.join(D,BX),'w',encoding=_R)as h:json.dump(Ao,h,ensure_ascii=_C)
	Ap=len(G)if G is not _A else 0;F=_A;S=[]
	if L<MIN_ROWS_FOR_SPLIT:F=f"データが {L} 行のため全データで学習。R² は訓練スコアのため楽観的な値です。";S.append({Z:'small_full_data',a:{BY:L}})
	elif An==_V:
		F=f"OOF (交差検証) で評価。"
		if L<SMALL_N_OOF_THRESH:F=f"少ないデータ（{L} 行）— "+F;S.append({Z:'oof_eval_small_n',a:{BY:L}})
		else:S.append({Z:'oof_eval',a:{}})
	elif Ap<20:F=f"検証セットが {Ap} 行と少なく R² が不安定な場合があります。100行以上推奨。";S.append({Z:'small_val_set',a:{'n_val':Ap}})
	if AR>0:BL=f"目的変数が欠損している {AR} 行を学習から除外しました。";F=F+_g+BL if F else BL;S.append({Z:'target_na_excluded',a:{'n':AR}})
	if AF>0:BM=OUTLIER_IQR_MULT if H else OUTLIER_IQR_QUICK;BN=f"Y 外れ値 {AF} 行を許容範囲内に補正しました（IQR×{BM}）。";F=F+_g+BN if F else BN;S.append({Z:'outliers_corrected',a:{'n':AF,'iqr_mult':BM}})
	if Ay:BO='Y 外れ値クリップは交差検証で信号を損なうと判定し非適用（正規の裾を温存）。';F=F+_g+BO if F else BO;S.append({Z:'winsorize_skipped_cv',a:{}})
	if A2>0:BP=f"重複行が{A2}件あります。評価が楽観的になる可能性があります。";F=F+_g+BP if F else BP;S.append({Z:'duplicate_rows',a:{'n':A2}})
	if d:BQ=f"カーディナリティが行数に対して高すぎるため除外した列: {', '.join(d)}";F=F+_g+BQ if F else BQ;S.append({Z:'cat_cols_dropped',a:{_i:d}})
	z,Aq,Ar=[],_A,[];As,AL=_A,[]
	try:BR=_sim_used_raw_columns(A3,A,AA,c,b,k,d,l,AS);BS=_sim_aggregate_importance(BE,BR,c,k,l,AS,P);z=_sim_build_slider_spec(A3,BR,BS,N);Bv=[A[_y]for A in z];BT=[A[_y]for A in z if A['kind']=='numeric'];Aq=_sim_build_y_hist(T);Ar=_sim_build_seed_rows(A3,A,Bv);As=_sim_build_neighbor_ref(A3,BT,BS);AL=_sim_build_corr_pairs(A3,BT);print(f"[Simulate] スライダー {len(z)} 列 / 相関ペア {len(AL)} 件",flush=_B)
	except Exception as f:print(f"[Simulate] UIデータの生成に失敗（SIMULATEは無効化）: {f}",flush=_B);z,Aq,Ar=[],_A,[];As,AL=_A,[]
	i={AQ:AK,'r2_raw':Ai,'rmse':round(Al,4)if Al is not _A else _A,'mae':round(Am,4)if Am is not _A else _A,'best_model':M,AD:u,'feature_importance':(BE or[])[:10],'slider_spec':z,'y_hist':Aq,'seed_rows':Ar,'neighbor_ref':As,'corr_pairs':AL,'eval_on':An,'train_rows':len(C),'val_rows':BH,_A3:A,'preset':AE,'data_warning':F,'data_warning_parts':S,'r2_interpretation':_r2_interpretation(AK),'r2_reference_only':bool(L<MIN_ROWS_FOR_SPLIT),'use_gp':u==_b,'gp_format':'pkl'if u==_b else _A,'scatter':BG,'y_range':[round(float(np.min(T)),4),round(float(np.max(T)),4)],'cat_columns':sorted(set(A[_P]for A in c)|set(b)),'cat_dropped_columns':d,'candidate_models':[{_G:D,AQ:round(float(C[0]),4)if np.isfinite(C[0])else _A,AD:C[2],BZ:D==M,Ba:round(_candidate_r2_std(C[3],C[4].get(_K)if C[4]else _A,V,B[A].values),4),_J:round(float(C[4].get(_J)),4)if C[4]and C[4].get(_J)is not _A else _A}for(D,C)in sorted(E.items(),key=lambda kv:kv[1][0]if np.isfinite(kv[1][0])else-np.inf,reverse=_B)]+([{_G:'MLP',AQ:_A,AD:_A,BZ:_C,Ba:_A,_J:_A,'skip_reason':A7,'skip_rows_needed':MLP_MIN_ROWS if A7==BW else _A}]if A7 else[])};_emit_progress(96,'saving_model');AM={B:A for(B,A)in E.items()if A[4]is not _A and A[4].get(_Q,_C)};Bw=[M]+sorted([A for A in AM if A!=M],key=lambda n:AM[n][0],reverse=_B);A0=_C;At=_C;j,AN=_A,_A
	for j in Bw:
		if j not in AM:continue
		AN,r,AO,Bx,A1=AM[j];By=_y_true_for(A1.get(_K,_L),T,W,m);AP,r,r,r=_fit_postprocess_params(Bx,By,I,T,Ax)
		if AO==_d:AC=_export_treg_blend(D,A,E,I,J,AP,(v,w),x,N,derived_recipe=P,cat_encoders_all=A4)
		elif AO==_X and A1 and(A1.get(_A5)or 0)>=2:
			AC=_export_treg_lgbm_bag(D,A,A1[_N],A1[_F],A1[_A5],I,J,AP,(v,w),x,N,derived_recipe=P,cat_encoders_all=A4)
			if AC:At=_B
			else:print('[TREG] lgbm_bag 失敗 → 単体LightGBMにフォールバック',flush=_B);AC=_export_treg(AO,D,A,I,J,AP,(v,w),x,N,derived_recipe=P,cat_encoders_all=A4)
		else:AC=_export_treg(AO,D,A,I,J,AP,(v,w),x,N,derived_recipe=P,cat_encoders_all=A4)
		if AC:
			A0=_B
			if j!=M:print(f"[TREG] 表示モデル({M})の書き出しに失敗 → デプロイは {j} (R²={AN:.4f}) を使用",flush=_B)
			break
	if not A0:print('[TREG] WARNING: デプロイ可能なモデルが存在しません',flush=_B)
	if At:
		Ao[AD]='lgbm_bag'
		with open(os.path.join(D,BX),'w',encoding=_R)as h:json.dump(Ao,h,ensure_ascii=_C)
	i['export_available']=A0;i['deployed_model']=j if A0 else _A;i['deployed_r2']=round(float(AN),4)if A0 and AN is not _A else _A;i['deploy_substituted']=bool(A0 and j!=M)
	if u==_d:print('[Python] Blend が最良のため、全サブモデルファイルを保持します',flush=_B)
	else:
		_clean_model_files(D,u,keep_lgbm_bag=At);BU=os.path.join(D,_AD)
		if os.path.exists(BU):os.remove(BU)
	try:
		if os.path.exists(Aa):shutil.rmtree(Aa)
		os.replace(D,Aa)
	except Exception as f:_error_exit('result_move_failed',detail=str(f))
	_emit_progress(100,'done');i=_sanitize_json(i);print(f"RESULT_JSON:{json.dumps(i,ensure_ascii=_C)}",flush=_B);print('[Python] 学習完了。',flush=_B)
if __name__=='__main__':asyncio.run(_run_main())