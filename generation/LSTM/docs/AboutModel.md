# LSTMを用いた分子生成モデル

詳細と実装について説明する。

## データの取得と前処理

分子構造をSMILESで表して系列モデルで扱うためにSMILESのアルファベット集合$\mathcal{X}$(ここで$\mathcal{X}$は文字列の取りうる範囲を表す集合で、離散的な集合とする。例: $\mathcal{X} = \{C, O, N, =, \dots \}$)を構築する。特に系列長を指定しない系列の集合は

$$
\mathcal{X}^* = \bigcup_{t=1}^{\infty} \mathcal{X}^T
$$

と表す。各SMILES系列$\bm{x} \in \mathcal{X}^*$から、識別モデル用のデータ$(\tilde{\bm{x}}, \tilde{\bm{y}}) \in \tilde{\mathcal{X}}^* \times \tilde{\mathcal{X}}^* $を作る。

SMILESのデータに対して、次のように識別モデル用に変換するための前処理を行う。

1. SMILES系列それぞれに開始記号\<sos>と終了記号\<eos>を付け加える。
2. 全ての系列の長さを揃えるために終了記号の後を空文字\<pad>で埋める（padding）
3. 全SMILES系列で使われる文字$\bar{\mathcal{X}} = \mathcal{X} \cup \{<sos>, <eos>, <pad> \}$を集計し、$\bar{\mathcal{X}}$と整数の対応表（辞書）を作る。

全ての系列の長さを揃えることで複数の系列に対して同時に計算を行え、予測や学習効率を上げることができる。
