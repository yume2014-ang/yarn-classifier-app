# 必要なライブラリをインポート
import streamlit as st
from PIL import Image

# アプリ背景用
import base64

# --- AIモデル関連ライブラリ ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
# ------------------------------------

# --- Grad-CAM 実装用ライブラリ ---
# import numpy as np
# import tensorflow as tf
import matplotlib.cm as cm
# from PIL import Image
# ------------------------------------

# --- クラス名をモデルの学習順に定義 ---
# モデルが学習した通りの順番で、素材名をリストにします
# CLASS_NAMES = ['ウール', 'アクリル', 'コットン', 'カシミヤ', 'シルク']
CLASS_NAMES = ['コットン', 'ウール']
# ------------------------------------------------

# --- GPU設定用 ---
@st.cache_resource(show_spinner=False)
def set_gpu():
    """
    GPUがあるか確認し、
    メモリ使用量を「必要な分だけ使う」設定にする
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU {len(gpus)}台を認識しました。")
        except RuntimeError as e:
            # プログラム起動後に設定を変えようとするとエラーになるため、その対策
            print(e)
    else:
        print("GPUは検出されませんでした（CPUで動作します）。")

    st.sidebar.write("システム情報")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        st.sidebar.success(f"🚀 GPUモード: {len(gpus)}台")
    else:
        st.sidebar.warning("🐢 CPUモード")
# ---------------------

# --- モデル読み込み(キャッシュ機能を使う) ---
# 
# @st.cache_resource：Streamlitの「デコレータ」と呼ばれる機能
#   ・この下に定義された関数（load_keras_model()）はアプリが起動した初回だけ実行
#   ・2回目以降の再実行時には、実行済みの結果（読み込んだ model オブジェクト）をキャッシュから再利用
# ------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model():
    """
    Kerasモデルを読み込みます。
    ファイルパスは適宜修正してください。
    """
    # --- モデルのファイル名を指定 ---
    model_path = 'yarn_material_model.keras'

    with st.spinner("AIモデルを準備しています。少々お待ちください... ⏳"):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return None

# --- 背景画像設定 ---
def set_overlay_bg(image_file):
    # 画像をBase64に変換
    with open(image_file, "rb") as f:
        data = f.read()
    b64_image = base64.b64encode(data).decode()
    
    # CSSを変更：linear-gradient（半透明の膜）を追加
    #   白いフィルター：rgba(255, 255, 255, 0.6)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: 
                /* 1層目：半透明の黒いフィルター(ダークモード風) */
                linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                /* 2層目：あなたの画像 */
                url("data:image/png;base64,{b64_image}");
            
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 画像の前処理 ---
def preprocess_image(image_pil):
    """
    PIL Image オブジェクトをモデルの入力形式に前処理します。
    """
    # --- モデルの入力サイズを指定 ---
    target_size = (224, 224) 
    # ------------------------------------------

    # 1. RGBA (透過あり) の場合、RGB (透過なし) に変換
    if image_pil.mode == 'RGBA':
        image_pil = image_pil.convert('RGB')
        
    # 2. 指定サイズにリサイズ
    image_resized = image_pil.resize(target_size)
    
    # 3. PIL Image を Numpy 配列に変換 (shape: (224, 224, 3))
    #    この時点での値の範囲は [0, 255]
    # image_array = np.array(image_resized) # <- 画像データの元の型を維持。通常の画像なら 整数（uint8, 0 から 255） になる
    # ↓ AIの計算で扱いやすいように、デフォルトで 小数（float32, 0.0 から 255.0）に変換
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)

    # 4. バッチ次元を追加 (shape: (1, 224, 224, 3))
    #    Kerasの model.predict() はバッチでの入力を期待するため
    expanded_array = np.expand_dims(image_array, axis=0)
    
    # 5. EfficientNet用の前処理を適用
    #    入力は [0, 255] のNumpy配列 (バッチ) を期待します
    #    出力は [-1, 1] にスケーリングされます
    preprocessed_array = preprocess_input(expanded_array.copy())
    
    return preprocessed_array

# --- Grad-CAM 実装用 --->
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation", pred_index=None):
    """
    指定した層(top_activation)の出力と、予測値の勾配を使ってヒートマップを作成する関数
    """
    # 1. モデルの入力から「ターゲット層の出力」と「モデルの最終出力」を出すサブモデルを作成
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. 勾配を計算
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)

        # ★エラー対策の修正ここから★
        # preds（予測結果）がリストで返ってきている場合、中身(テンソル)を取り出す
        if isinstance(preds, list):
            preds = preds[0]
        # last_conv_layer_output も念のため同様に処理
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]
        # ★エラー対策の修正ここまで★

        if pred_index is None:
            pred_index = tf.argmax(preds[0]) # 予測確率が一番高いクラスを自動選択
        class_channel = preds[:, pred_index]

    # 3. ターゲット層の出力に対する、予測クラスの勾配を取得
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. 各チャネルの重要度（重み）を計算（Global Average Poolingのような処理）
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. 重要度を重みとして、ターゲット層の出力（7x7x1280）にかけて足し合わせる
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. ヒートマップを正規化（0〜1の範囲に収め、ReLUを通してマイナス値を消す）
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_superimposed_image(heatmap, original_img_pil, alpha=0.4):
    """
    ヒートマップを元の画像サイズに拡大し、重ね合わせる関数
    original_img_pil: PIL形式の元画像
    """
    # ヒートマップを0-255の範囲の8bit整数に変換
    heatmap = np.uint8(255 * heatmap)

    # カラーマップ（jet）を適用してRGB画像にする
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # ヒートマップを画像化し、元画像のサイズ(224x224)にリサイズ
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img_pil.width, original_img_pil.height))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 元画像も配列に変換
    original_img_array = tf.keras.preprocessing.image.img_to_array(original_img_pil)

    # 画像を重ね合わせる (alphaでヒートマップの透明度調整)
    superimposed_img = jet_heatmap * alpha + original_img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

# <--- Grad-CAM 実装用関数 ---

# --- HTMLを使って色付きのバーを表示 ---
def custom_progress_bar(probability):
    # 1. 確率に応じて色を決める（ここは自由に変えられます）
    if probability >= 0.8:
        bar_color = "#28a745" # 緑色（高確信度）
    elif probability >= 0.5:
        bar_color = "#ffc107" # 黄色（中確信度）
    else:
        bar_color = "#dc3545" # 赤色（低確信度）

    # 2. パーセント表示にする
    percent = probability * 100

    # 3. HTMLでバーを描画
    # 背景は薄いグレー、その上に色のついたバーを重ねています
    st.markdown(f"""
    <div style="width: 100%; background-color: #f0f2f6; border-radius: 5px; height: 12px; margin-top: 7px;">
        <div style="width: {percent}%; background-color: {bar_color}; height: 100%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)


# --- メイン処理 ---
def main():
    # アプリ起動時に一度だけ実行
    set_gpu()
    model = load_keras_model()  # モデルを読み込む

    # 背景画像設定
    try:
        set_overlay_bg("bg_YarnClassAI.png") 
    except FileNotFoundError:
        st.warning("画像ファイルが見つかりません")

    # 中央に配置するためのカラム
    col1, col2, col3 = st.columns([1, 5, 1]) # [左のスペーサー, 真ん中, 右のスペーサー]

    # 真ん中のカラム (col2) のコンテキスト内
    with col2:

        # 1. アプリのタイトルを設定
        st.title('🧶 :orange[_Wool_] or :orange[_Cotton_] 🧶')

        # 2. 説明文を追加
        st.subheader('毛糸素材分類AIアプリへようこそ！')
        st.divider()
        st.write("毛糸の画像をアップロードすると素材を判定します")

        # 3. 画像アップローダーを設置
        uploaded_file = st.file_uploader("毛糸の画像を選択してください...", type=['jpg', 'jpeg', 'png'])

        # 4. ファイルがアップロードされた場合の処理
        if uploaded_file is not None:
            try:
                # アップロードされた画像を読み込む
                image = Image.open(uploaded_file)

                # 画像を表示
                # st.image(image, caption='アップロードされた画像', width=400) 
                st.image(image, caption='アップロードされた画像', width="stretch")
                
                # ボタンを中央に配置するためのカラム (col2の中でネスト)
                btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

                # 真ん中のボタン用カラム (btn_col2) のコンテキスト内
                # btn_col2.button(...) を使ってボタンを配置し、押されたかどうかの状態を受け取る
                button_pressed = btn_col2.button('素材を判定する', width="stretch")

                # 分類処理 (ボタンが押されたら実行)
                if button_pressed:
                    if model is None:
                        st.error("モデルが読み込まれていないため、判定できません。")
                    else:
                        with st.spinner('AIが画像を分析中です... 少々お待ちください...⏳'):
                            
                            # 画像の前処理
                            processed_image = preprocess_image(image)
                            
                            # モデルによる予測の実行
                            # predictions は各クラス（素材）の確率のリスト（Numpy配列）を返す
                            predictions = model.predict(processed_image)
                            
                            # 予測結果の解析
                            # predictions に全クラスの確率配列が入っている
                            # 予測結果にSoftmaxを適用して確率値（0.0〜1.0）に変換
                            # .flatten() を使用して一次元配列（リスト）にしておく
                            # ここで all_probabilities に [0.1, 0.8, 0.1] のような確率値が入る
                            probabilities_tensor = tf.nn.softmax(predictions)
                            all_probabilities = probabilities_tensor.numpy().flatten()
                            
                            # (確率, クラス名) のタプルのリストを作成
                            prob_list = list(zip(CLASS_NAMES, all_probabilities))
                            
                            # 確率 (x[1]) を基準に降順 (reverse=True) でソート
                            sorted_prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)

                            # ================
                            # Grad-CAMの実行
                            # ================
                            # ヒートマップを作成
                            # last_conv_layer_nameは "top_activation" を指定
                            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name="top_activation")

                            # 元画像に重ね合わせ
                            superimposed_img = create_superimposed_image(heatmap, image, alpha=0.4)

                        # おまけ: 分類成功のエフェクト
                        st.snow()
                        # st.balloons()

                        st.success('判定が完了しました！')
                        st.subheader('🎉 判定結果')

                        # 信頼度を確率バーで表示
                        # ソートされたリストをループ処理
                        for i, (class_name, probability) in enumerate(sorted_prob_list):
                            
                            # --- 🥇 1位のみリッチに表示 ---
                            if i == 0:
                                # 特別感のあるコンテナを作る
                                with st.container():
                                    # タイトルと確率を大きな指標（Metric）で表示
                                    st.metric(label="予測結果：", 
                                            value=f"🥇 {class_name}　{probability:.1%}", 
                                            label_visibility="collapsed",
                                            border=True)
                                    
                                # 余白を入れる
                                st.write("")
                                st.write("")

                            # --- リスト形式で表示 ---
                            # カラムで横並びにする（比率は 3:5:2 くらいが綺麗）
                            r_col1, r_col2, r_col3 = st.columns([3, 5, 2])
                            with r_col1:
                                # 名前（左寄せ）
                                st.write(f"**{i+1}. {class_name}**")
                            
                            with r_col2:
                                # プログレスバー（中央）
                                # st.progress(float(probability))
                                custom_progress_bar(probability)
                                
                            with r_col3:
                                # 数値（右側）
                                st.write(f"{probability:.1%}")
                        
                        st.divider()

                        # ==========================================
                        # Grad-CAMの表示
                        # ==========================================
                        st.write("### 👁️ AIの注目領域 (by Grad-CAM)")

                        # 左右に並べて比較
                        gcam_col1, gcam_col2 = st.columns(2)
                        with gcam_col1:
                            st.image(image, caption="元画像", width="stretch")
                        with gcam_col2:
                            st.image(superimposed_img, caption="判定の根拠(赤=注目)", width="stretch")

                else:
                    # ボタンが押される前のデフォルトのメッセージ
                    st.warning("（ボタンを押して判定を開始してください）")

            except Exception as e:
                st.error(f"画像の読み込み中にエラーが発生しました: {e}")

        else:
            # 背景画像で見にくいので st.info -> st.warning に変更
            st.warning("⬆️ 上のボタンから画像ファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
