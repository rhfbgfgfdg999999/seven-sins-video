# 复刻提示词

把下面整段提示词原样发给任意 AI 编程助手（Claude / ChatGPT / ZCode 等）：AI 会**先从 GitHub 完整部署代码，自检通过后再渲染出全部 7 条成品视频**。也可以直接把本文件发给 AI。

---

```text
请帮我完整部署并复刻这套「七宗罪」七部曲抖音短视频生成器。
代码仓库：https://github.com/rhfbgfgfdg999999/seven-sins-video
（MIT 协议；全部画面与音效由程序生成，不使用任何外部图片 / 视频 / 音频素材）

【第一步：完整部署代码（必须先做，先不渲染）】
1. 克隆仓库：git clone https://github.com/rhfbgfgfdg999999/seven-sins-video.git
2. 安装依赖：pip install -r requirements.txt（numpy scipy opencv-python pillow），
   并确认命令行能运行 ffmpeg -version
3. 准备 Noto CJK 字体（思源黑体/宋体 .ttc）：Linux 默认查找
   /usr/share/fonts/opentype/noto/，其他平台设置环境变量 SINS_FONT_DIR 指向字体目录
4. 核对工程结构：scripts/（make.py 渲染、sfx.py 音效、sins_data.py 全部文案等）、
   web/（index.html 播放器 + videos/ 成品）、references/copy.md、LICENSE（MIT）
5. 部署自检：cd scripts && python3 make.py pride --preview 0.5,5.3,9
   —— 能秒级输出 3 张静帧（work/pv_*.png）即部署成功

【第二步：出片（部署自检通过后再执行）】
1. 全部渲染：cd scripts && ./render_all.sh
   —— 7 条并行（每条约 3 分钟），程序合成的音效自动混入，输出到 scripts/out/
2. 单条渲染：python3 make.py pride --out work/pride.mp4
3. 把成品放进 web/videos/，双击 web/index.html 自动连播；页面可一键复制每条的
   发布标题与话题标签
4. 最终交付：01_傲慢 02_嫉妒 03_暴怒 04_懒惰 05_贪婪 06_暴食 07_色欲，
   共 7 条 1920×1080、30fps、18 秒成品，可直接发布抖音

【这套程序的细节参考（排查问题时用）】
- 画面：粒子人头（SDF 隐式建模 → 表面点采样 → 加性双线性点溅射渲染），
  暗场 + 主题色辉光；每罪专属标志物（傲慢=王冠、嫉妒=毒蛇、暴怒=双角、
  贪婪=金币雨、暴食=旋涡、色欲=花瓣，懒惰主题另配）
- 时间轴按 ~90.25 BPM 节拍网格（BEAT=0.6648s，B0=-0.055s，DROP 第 8 拍 ≈5.26s）：
  阵容 → 推近闭眼 → riser → 屏息 → DROP+三次硬切 → 变身 → 字卡（罪名+拉丁文+
  罗马数字+对立美德）→ 六个四字词逐拍弹出 → 冲击波 → 淡出
- 文案字段集中在 scripts/sins_data.py 的 SINS 列表：cn/latin/roman/nth、
  virtue_cn/virtue_lat、lineA/lineB（"kw" 片段主题色高亮）、words（六个四字词）、
  title（发布标题）、theme/accent/kw（配色）、pose/mouth/eyes；改字段即可换主题
- 音效由 scripts/sfx.py 程序合成（每拍底鼓、DROP 低音 boom、呼啸 riser、切镜
  whoosh、词 zap、和弦 stab、冲击波 sweep、hi-hat/clap），与画面同一节拍网格，
  make.py 自动混入；想让视觉特效跟随自己的音乐：python3 audio_fx.py 音乐.wav 后再渲染
- 渲染确定性（固定随机种子），同一命令两次渲染画面一致

【质量要求】
- 部署阶段不要改动任何渲染逻辑与文案；出片阶段如遇报错，先按仓库 README
  的「使用教程」排查
- 严格按顺序执行：先完成第一步部署自检，再进入第二步出片
```

---

> 💡 想换主题（比如「七美德」「七种孤独」）：在提示词后面加一句
> 「主题改成 XX，按同样的结构重新设计七个罪的文案、配色与标志物」即可。
