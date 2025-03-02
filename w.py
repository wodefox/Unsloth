import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import matplotlib.font_manager as fm
import requests
import warnings
from tkinter import font as tkfont
import sv_ttk  # 导入Sun Valley主题包，需要先安装: pip install sv-ttk

# 设置requests超时时间和重试次数
requests.adapters.DEFAULT_RETRIES = 3
requests.DEFAULT_TIMEOUT = 30

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FineTuningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unsloth 模型微调工具")
        self.root.geometry("1100x800")
        self.root.minsize(900, 700)  # 设置最小窗口大小
        
        # 设置字体
        self.default_font = tkfont.Font(family='Microsoft YaHei UI', size=10)
        self.title_font = tkfont.Font(family='Microsoft YaHei UI', size=11, weight='bold')
        
        # 应用Sun Valley主题
        sv_ttk.set_theme("light")
        self.theme_mode = tk.StringVar(value="light")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置根窗口的行列权重，使界面可以随窗口调整大小
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)
        
        # 网络设置
        self.proxy = tk.StringVar()
        self.timeout = tk.StringVar(value="30")
        self.max_retries = tk.StringVar(value="3")
        self.offline_mode = tk.BooleanVar(value=False)
        self.local_model_dir = tk.StringVar()
        
        # 模型映射字典
        self.model_mapping = {
            "Llama": {
                "1B": "unsloth/llama-3.1-1b",
                "3B": "unsloth/llama-3.2-3b",
                "7B": "meta-llama/Llama-2-7b-hf",
                "13B": "meta-llama/Llama-2-13b-hf",
                "70B": "meta-llama/Llama-2-70b-hf"
            },
            "Qwen": {
                "1B": "unsloth/Qwen2.5-0.5B",
                "3B": "unsloth/Qwen2.5-3B",
                "7B": "unsloth/Qwen2.5-7B",
                "14B": "unsloth/Qwen2.5-14B",
                "32B": "unsloth/Qwen2.5-32B",
                "72B": "unsloth/Qwen2.5-72B"
            },
            "Mistral": {
                "7B": "mistralai/Mistral-7B-v0.1",
                "8B": "mistralai/Mistral-8x7B-v0.1"
            },
            "Phi": {
                "1B": "microsoft/phi-1",
                "2B": "microsoft/phi-2"
            }
        }

        # 创建顶部工具栏
        self.create_toolbar()
        
        # 左侧面板 - 使用Notebook创建选项卡界面
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.left_panel.columnconfigure(0, weight=1)

        # 创建选项卡
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 基本设置选项卡
        self.basic_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.basic_tab, text="基本设置")
        self.basic_tab.columnconfigure(0, weight=1)
        
        # 高级设置选项卡
        self.advanced_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.advanced_tab, text="高级设置")
        self.advanced_tab.columnconfigure(0, weight=1)
        
        # 网络设置选项卡
        self.network_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.network_tab, text="网络设置")
        self.network_tab.columnconfigure(0, weight=1)
        
        # 模型选择框架
        model_frame = ttk.LabelFrame(self.basic_tab, text="模型选择", padding="8")
        model_frame.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        model_frame.columnconfigure(1, weight=1)
        model_frame.columnconfigure(3, weight=1)
        
        # 模型系列选择
        ttk.Label(model_frame, text="模型系列:").grid(row=0, column=0, sticky=tk.W)
        self.model_family = tk.StringVar(value="Llama")
        model_family_combo = ttk.Combobox(model_frame, textvariable=self.model_family, width=15)
        model_family_combo["values"] = ("Llama", "Qwen", "Mistral", "Phi", "其他")
        model_family_combo.grid(row=0, column=1, sticky=tk.W)
        model_family_combo.bind("<<ComboboxSelected>>", self.update_model_options)
        
        # 模型大小选择
        ttk.Label(model_frame, text="模型大小:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.model_size = tk.StringVar(value="7B")
        self.model_size_combo = ttk.Combobox(model_frame, textvariable=self.model_size, width=10)
        self.model_size_combo["values"] = ("1B", "3B", "7B", "13B", "70B")
        self.model_size_combo.grid(row=0, column=3, sticky=tk.W)
        self.model_size_combo.bind("<<ComboboxSelected>>", self.update_model_name)
        
        # 模型名称
        ttk.Label(model_frame, text="完整模型名称:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.model_var = tk.StringVar(value="meta-llama/Llama-2-7b-hf")
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=50)
        self.model_entry.grid(row=1, column=1, columnspan=3, sticky=tk.W+tk.E, pady=(5, 0))

        # 训练数据选择
        data_frame = ttk.LabelFrame(self.basic_tab, text="数据设置", padding="8")
        data_frame.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        data_frame.columnconfigure(1, weight=1)
        
        ttk.Label(data_frame, text="训练数据:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_path = tk.StringVar()
        data_entry = ttk.Entry(data_frame, textvariable=self.data_path)
        data_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        data_btn = ttk.Button(data_frame, text="选择文件", command=self.select_data, width=10)
        data_btn.grid(row=0, column=2, padx=5, pady=5)
        self.create_tooltip(data_entry, "选择用于微调模型的训练数据文件，通常为JSON格式")

        # 模型保存路径
        ttk.Label(data_frame, text="保存路径:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_path = tk.StringVar(value="./output")
        save_entry = ttk.Entry(data_frame, textvariable=self.save_path)
        save_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        save_btn = ttk.Button(data_frame, text="选择目录", command=self.select_save_path, width=10)
        save_btn.grid(row=1, column=2, padx=5, pady=5)
        self.create_tooltip(save_entry, "设置微调后模型的保存目录")

        # 训练参数设置
        self.create_training_params()
        
        # 控制按钮
        self.create_control_buttons()

        # 高级选项
        self.create_advanced_options()

        # 创建状态面板
        status_frame = ttk.LabelFrame(self.basic_tab, text="训练状态", padding="8")
        status_frame.grid(row=4, column=0, pady=5, sticky=tk.W+tk.E+tk.N+tk.S)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        
        # 进度条
        progress_frame = ttk.Frame(status_frame)
        progress_frame.grid(row=0, column=0, sticky=tk.W+tk.E, pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5)

        # 训练日志
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # 右侧面板（训练可视化）
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        self.setup_visualization()

        self.training_active = False

    def create_training_params(self):
        params_frame = ttk.LabelFrame(self.basic_tab, text="训练参数", padding="8")
        params_frame.grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)

        # 基础参数 - 使用网格布局并添加适当的间距
        lr_label = ttk.Label(params_frame, text="学习率:")
        lr_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="2e-5")
        lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lr_entry, "模型学习率，通常在1e-5到5e-5之间")

        batch_label = ttk.Label(params_frame, text="Batch Size:")
        batch_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value="4")
        batch_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10)
        batch_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(batch_entry, "每批处理的样本数量，根据显存大小调整")

        epochs_label = ttk.Label(params_frame, text="训练轮数:")
        epochs_label.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="3")
        epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(epochs_entry, "完整训练数据集的迭代次数")

        length_label = ttk.Label(params_frame, text="最大长度:")
        length_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_length_var = tk.StringVar(value="512")
        length_entry = ttk.Entry(params_frame, textvariable=self.max_length_var, width=10)
        length_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(length_entry, "输入序列的最大长度，影响显存使用量")
    def validate_inputs(self):
        """验证输入参数的有效性"""
        try:
            if not self.model_var.get().strip():
                raise ValueError("请输入基础模型名称")
                
            if not self.data_path.get():
                raise ValueError("请选择训练数据文件")
                
            if not os.path.exists(self.data_path.get()):
                raise ValueError("训练数据文件不存在")
                
            # 验证数值参数
            lr = float(self.lr_var.get())
            if lr <= 0 or lr >= 1:
                raise ValueError("学习率必须在0到1之间")
                
            batch_size = int(self.batch_size_var.get())
            if batch_size <= 0:
                raise ValueError("Batch Size必须大于0")
                
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError("训练轮数必须大于0")
                
            max_length = int(self.max_length_var.get())
            if max_length <= 0:
                raise ValueError("最大长度必须大于0")
                
            if self.use_lora.get():
                lora_rank = int(self.lora_rank.get())
                if lora_rank <= 0:
                    raise ValueError("LoRA rank必须大于0")
                    
            grad_accum = int(self.grad_accum.get())
            if grad_accum <= 0:
                raise ValueError("梯度累积步数必须大于0")
                
            return True
            
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return False
    def create_toolbar(self):
        """创建顶部工具栏"""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # 主题切换按钮
        theme_frame = ttk.Frame(toolbar)
        theme_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(theme_frame, text="主题:").pack(side=tk.LEFT, padx=(0, 5))
        theme_light = ttk.Radiobutton(theme_frame, text="浅色", value="light", variable=self.theme_mode, command=self.toggle_theme)
        theme_light.pack(side=tk.LEFT, padx=5)
        theme_dark = ttk.Radiobutton(theme_frame, text="深色", value="dark", variable=self.theme_mode, command=self.toggle_theme)
        theme_dark.pack(side=tk.LEFT, padx=5)
        
        # 版本信息
        version_label = ttk.Label(toolbar, text="Unsloth 微调工具 v1.0")
        version_label.pack(side=tk.LEFT, padx=10)
        
        # 分隔线
        separator = ttk.Separator(self.root, orient="horizontal")
        separator.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        
        # 调整主窗口的行配置
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        
        # 将主框架移到第三行
        self.main_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def toggle_theme(self):
        """切换主题模式"""
        theme = self.theme_mode.get()
        sv_ttk.set_theme(theme)
        
    def create_tooltip(self, widget, text):
        """为控件创建工具提示"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # 创建工具提示窗口
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, justify=tk.LEFT,
                             background="#ffffe0", relief="solid", borderwidth=1,
                             font=("Microsoft YaHei UI", 9))
            label.pack(padx=3, pady=3)
            
        def leave(event):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def create_advanced_options(self):
        adv_frame = ttk.LabelFrame(self.advanced_tab, text="高级训练选项", padding="8")
        adv_frame.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        adv_frame.columnconfigure(1, weight=1)
        adv_frame.columnconfigure(3, weight=1)

        # LoRA配置
        self.use_lora = tk.BooleanVar(value=True)
        lora_check = ttk.Checkbutton(adv_frame, text="使用LoRA", variable=self.use_lora)
        lora_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lora_check, "启用LoRA参数高效微调，减少显存占用")

        ttk.Label(adv_frame, text="LoRA rank:").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.lora_rank = tk.StringVar(value="8")
        lora_rank_entry = ttk.Entry(adv_frame, textvariable=self.lora_rank, width=10)
        lora_rank_entry.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lora_rank_entry, "LoRA的秩，越大效果越好但显存占用越多")

        # 混合精度训练
        self.use_fp16 = tk.BooleanVar(value=True)
        fp16_check = ttk.Checkbutton(adv_frame, text="使用FP16", variable=self.use_fp16)
        fp16_check.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(fp16_check, "启用混合精度训练，加速训练并减少显存占用")

        # 梯度累积
        ttk.Label(adv_frame, text="梯度累积步数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.grad_accum = tk.StringVar(value="4")
        grad_accum_entry = ttk.Entry(adv_frame, textvariable=self.grad_accum, width=10)
        grad_accum_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(grad_accum_entry, "梯度累积步数，可以模拟更大的批量大小")
        
        # 优化器选项
        ttk.Label(adv_frame, text="优化器:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.optimizer_var = tk.StringVar(value="adamw_8bit")
        optimizer_combo = ttk.Combobox(adv_frame, textvariable=self.optimizer_var, width=12)
        optimizer_combo["values"] = ("adamw_8bit", "adamw_torch", "adam", "sgd")
        optimizer_combo.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(optimizer_combo, "选择优化器类型，adamw_8bit为节省显存的推荐选项")
        
        # 学习率调度器
        ttk.Label(adv_frame, text="学习率调度器:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_scheduler = tk.StringVar(value="linear")
        lr_scheduler_combo = ttk.Combobox(adv_frame, textvariable=self.lr_scheduler, width=12)
        lr_scheduler_combo["values"] = ("linear", "cosine", "cosine_with_restarts", "polynomial")
        lr_scheduler_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lr_scheduler_combo, "学习率变化策略，linear为线性衰减，cosine为余弦衰减")
        
        # 权重衰减
        ttk.Label(adv_frame, text="权重衰减:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.weight_decay = tk.StringVar(value="0.01")
        weight_decay_entry = ttk.Entry(adv_frame, textvariable=self.weight_decay, width=10)
        weight_decay_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(weight_decay_entry, "权重正则化系数，用于防止过拟合")
        
        # 数据处理选项
        ttk.Label(adv_frame, text="数据处理线程:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_proc = tk.StringVar(value="2")
        num_proc_entry = ttk.Entry(adv_frame, textvariable=self.num_proc, width=10)
        num_proc_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(num_proc_entry, "数据预处理的并行线程数，根据CPU核心数调整")
        
        # 序列打包选项
        self.use_packing = tk.BooleanVar(value=False)
        packing_check = ttk.Checkbutton(adv_frame, text="使用序列打包(加速短序列训练)", variable=self.use_packing)
        packing_check.grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(packing_check, "将多个短序列打包成一个长序列，提高训练效率")
        

        
        # 网络设置选项 - 移动到网络设置选项卡
        net_frame = ttk.LabelFrame(self.network_tab, text="网络连接设置", padding="8")
        net_frame.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        net_frame.columnconfigure(1, weight=1)
        
        # 离线模式
        offline_check = ttk.Checkbutton(net_frame, text="离线模式(使用本地模型)", variable=self.offline_mode)
        offline_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(offline_check, "启用后将使用本地模型，不从网络下载")
        
        # 本地模型目录
        ttk.Label(net_frame, text="本地模型目录:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        model_dir_entry = ttk.Entry(net_frame, textvariable=self.local_model_dir)
        model_dir_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        model_btn_frame = ttk.Frame(net_frame)
        model_btn_frame.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Button(model_btn_frame, text="浏览", command=self.select_model_dir, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_btn_frame, text="扫描模型", command=self.scan_local_models, width=8).pack(side=tk.LEFT, padx=2)
        
        # 代理设置
        proxy_frame = ttk.LabelFrame(self.network_tab, text="代理设置", padding="8")
        proxy_frame.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        proxy_frame.columnconfigure(1, weight=1)
        
        ttk.Label(proxy_frame, text="HTTP代理:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        proxy_entry = ttk.Entry(proxy_frame, textvariable=self.proxy)
        proxy_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.create_tooltip(proxy_entry, "设置HTTP代理，格式: http://host:port")
        
        # 超时设置
        timeout_frame = ttk.LabelFrame(self.network_tab, text="连接设置", padding="8")
        timeout_frame.grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        timeout_frame.columnconfigure(1, weight=1)
        timeout_frame.columnconfigure(3, weight=1)
        
        ttk.Label(timeout_frame, text="连接超时(秒):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        timeout_entry = ttk.Entry(timeout_frame, textvariable=self.timeout, width=10)
        timeout_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(timeout_entry, "设置网络请求超时时间，单位为秒")
        
        ttk.Label(timeout_frame, text="最大重试次数:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        retries_entry = ttk.Entry(timeout_frame, textvariable=self.max_retries, width=10)
        retries_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(retries_entry, "设置网络请求失败后的最大重试次数")
        


    def create_control_buttons(self):
        btn_frame = ttk.Frame(self.basic_tab)
        btn_frame.grid(row=3, column=0, pady=10, sticky=tk.W+tk.E)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        btn_frame.columnconfigure(3, weight=1)
        btn_frame.columnconfigure(4, weight=1)

        start_btn = ttk.Button(btn_frame, text="开始训练", command=self.start_training, width=12)
        start_btn.grid(row=0, column=0, padx=5, pady=5)
        self.create_tooltip(start_btn, "开始模型微调训练过程")
        
        pause_btn = ttk.Button(btn_frame, text="暂停训练", command=self.pause_training, width=12)
        pause_btn.grid(row=0, column=1, padx=5, pady=5)
        self.create_tooltip(pause_btn, "暂停当前训练过程")
        
        stop_btn = ttk.Button(btn_frame, text="停止训练", command=self.stop_training, width=12)
        stop_btn.grid(row=0, column=2, padx=5, pady=5)
        self.create_tooltip(stop_btn, "完全停止训练过程并重置进度")
        
        export_btn = ttk.Button(btn_frame, text="导出模型", command=self.export_model, width=12)
        export_btn.grid(row=0, column=3, padx=5, pady=5)
        self.create_tooltip(export_btn, "将训练好的模型导出为可部署格式")
        
        import_btn = ttk.Button(btn_frame, text="导入模型", command=self.import_model, width=12)
        import_btn.grid(row=0, column=4, padx=5, pady=5)
        self.create_tooltip(import_btn, "导入已有的模型配置和权重")

    def setup_visualization(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax1.set_title('训练损失')
        self.ax2.set_title('学习率')
        self.train_losses = []
        self.learning_rates = []

    def select_data(self):
        filename = filedialog.askopenfilename(
            title="选择训练数据",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.data_path.set(filename)

    def select_save_path(self):
        directory = filedialog.askdirectory(title="选择保存目录")
        if directory:
            self.save_path.set(directory)

    def update_visualization(self, loss, lr):
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(self.train_losses)
        # 使用支持中文的字体
        font_props = matplotlib.font_manager.FontProperties(family='SimHei')
        self.ax1.set_title('训练损失', fontproperties=font_props)
        
        self.ax2.plot(self.learning_rates)
        self.ax2.set_title('学习率', fontproperties=font_props)
        self.canvas.draw()
        self.root.update()
    def start_training(self):
        if not self.training_active:
            if self.validate_inputs():
                self.training_active = True
                Thread(target=self.training_process, daemon=True).start()
    def pause_training(self):
        if self.training_active:
            self.training_active = False
            self.log_text.insert(tk.END, "训练已暂停\n")
    def stop_training(self):
        self.training_active = False
        self.progress['value'] = 0
        self.log_text.insert(tk.END, "训练已停止\n")
    def import_model(self):
        """导入已有的模型"""
        try:
            # 选择模型目录
            model_dir = filedialog.askdirectory(title="选择模型目录")
            if not model_dir:
                return
                
            # 检查是否存在配置文件
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"在所选目录中找不到config.json文件")
                
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.log_text.insert(tk.END, f"正在导入模型配置...\n")
            
            # 更新界面参数
            if "base_model" in config:
                self.model_var.set(config["base_model"])
            if "learning_rate" in config:
                self.lr_var.set(config["learning_rate"])
            if "batch_size" in config:
                self.batch_size_var.set(config["batch_size"])
            if "epochs" in config:
                self.epochs_var.set(config["epochs"])
            if "max_length" in config:
                self.max_length_var.set(config["max_length"])
            if "use_lora" in config:
                self.use_lora.set(config["use_lora"])
            if "lora_rank" in config:
                self.lora_rank.set(config["lora_rank"])
            if "use_fp16" in config:
                self.use_fp16.set(config["use_fp16"])
            if "gradient_accumulation_steps" in config:
                self.grad_accum.set(config["gradient_accumulation_steps"])
                
            # 验证模型文件是否存在
            from transformers import AutoTokenizer
            try:
                # 尝试加载tokenizer以验证模型
                tokenizer_path = os.path.join(model_dir, "tokenizer")
                if os.path.exists(tokenizer_path):
                    AutoTokenizer.from_pretrained(tokenizer_path)
                    self.log_text.insert(tk.END, "成功验证tokenizer\n")
            except Exception as e:
                self.log_text.insert(tk.END, f"警告: tokenizer验证失败: {str(e)}\n")
                
            # 设置保存路径为导入模型的父目录
            parent_dir = os.path.dirname(model_dir)
            self.save_path.set(parent_dir)
            
            self.log_text.insert(tk.END, "模型导入成功！可以开始训练或导出模型。\n")
            messagebox.showinfo("成功", "模型配置已成功导入！")
            
        except Exception as e:
            self.log_text.insert(tk.END, f"导入失败: {str(e)}\n")
            messagebox.showerror("错误", f"导入失败: {str(e)}")
            
    def scan_local_models(self):
        """扫描本地模型目录并更新模型列表"""
        try:
            model_dir = self.local_model_dir.get()
            if not model_dir or not os.path.exists(model_dir):
                messagebox.showerror("错误", "请选择有效的本地模型目录")
                return

            self.log_text.insert(tk.END, f"正在扫描本地模型目录: {model_dir}...\n")
            
            # 扫描目录下的所有子文件夹
            model_folders = []
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含模型文件
                    config_path = os.path.join(item_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                # 获取模型信息
                                model_info = {
                                    'path': item_path,
                                    'name': item,
                                    'config': config,
                                    'base_model': config.get('base_model', '未知'),
                                    'params': self._get_model_params(config),
                                    'architecture': config.get('architectures', ['未知'])[0],
                                    'last_modified': time.strftime('%Y-%m-%d %H:%M:%S',
                                                                   time.localtime(os.path.getmtime(item_path)))
                                }
                                model_folders.append(model_info)
                        except Exception as e:
                            self.log_text.insert(tk.END, f"读取模型 {item} 配置失败: {str(e)}\n")

            if not model_folders:
                self.log_text.insert(tk.END, "未找到有效的模型文件夹\n")
                return

            # 创建模型选择对话框
            model_window = tk.Toplevel(self.root)
            model_window.title("选择本地模型")
            model_window.geometry("800x600")
            model_window.transient(self.root)
            model_window.grab_set()
            
            # 创建搜索框
            search_frame = ttk.Frame(model_window)
            search_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(search_frame, text="搜索模型:").pack(side=tk.LEFT)
            search_var = tk.StringVar()
            search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
            search_entry.pack(side=tk.LEFT, padx=5)
            
            # 创建表格显示模型信息
            columns = ('名称', '基础模型', '参数量', '架构', '最后修改时间')
            tree = ttk.Treeview(model_window, columns=columns, show='headings', height=15)
            
            # 设置列标题
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(model_window, orient=tk.VERTICAL, command=tree.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.configure(yscrollcommand=scrollbar.set)
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 存储模型信息的字典
            model_info_dict = {}
            
            def update_tree(search_text=''):
                tree.delete(*tree.get_children())
                for model in model_folders:
                    if search_text.lower() in model['name'].lower() or \
                       search_text.lower() in model['base_model'].lower():
                        item = tree.insert('', tk.END, values=(
                            model['name'],
                            model['base_model'],
                            model['params'],
                            model['architecture'],
                            model['last_modified']
                        ))
                        model_info_dict[item] = model
            
            def on_search(*args):
                update_tree(search_var.get())
            
            search_var.trace('w', on_search)
            
            def select_model():
                selection = tree.selection()
                if selection:
                    item = selection[0]
                    if item in model_info_dict:
                        selected_model = model_info_dict[item]
                        config = selected_model['config']
                        
                        # 更新界面参数
                        if "base_model" in config:
                            self.model_var.set(config["base_model"])
                            # 根据base_model自动设置模型系列和大小
                            for family, sizes in self.model_mapping.items():
                                for size, model_name in sizes.items():
                                    if model_name == config["base_model"]:
                                        self.model_family.set(family)
                                        self.update_model_options()
                                        self.model_size.set(size)
                                        break
                        
                        # 更新其他训练参数
                        if "learning_rate" in config:
                            self.lr_var.set(config["learning_rate"])
                        if "batch_size" in config:
                            self.batch_size_var.set(config["batch_size"])
                        if "epochs" in config:
                            self.epochs_var.set(config["epochs"])
                        if "max_length" in config:
                            self.max_length_var.set(config["max_length"])
                        if "use_lora" in config:
                            self.use_lora.set(config["use_lora"])
                        if "lora_rank" in config:
                            self.lora_rank.set(config["lora_rank"])
                        
                        self.log_text.insert(tk.END, f"已加载模型配置: {selected_model['path']}\n")
                        model_window.destroy()
                    else:
                        messagebox.showerror("错误", "无法加载所选模型的配置信息")
                else:
                    messagebox.showerror("错误", "请先选择一个模型")
            
            # 添加按钮框架
            button_frame = ttk.Frame(model_window)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="选择模型", command=select_model).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="取消", command=model_window.destroy).pack(side=tk.LEFT, padx=10)
            
            # 初始显示所有模型
            update_tree()
            
            # 等待窗口关闭
            self.root.wait_window(model_window)
            
        except Exception as e:
            self.log_text.insert(tk.END, f"扫描模型目录失败: {str(e)}\n")
            messagebox.showerror("错误", f"扫描模型目录失败: {str(e)}")
    
    def _get_model_params(self, config):
        """从模型配置中获取参数量"""
        try:
            # 尝试从不同的配置字段中获取参数量信息
            if 'n_parameters' in config:
                params = config['n_parameters']
            elif 'num_parameters' in config:
                params = config['num_parameters']
            else:
                # 估算参数量
                hidden_size = config.get('hidden_size', 0)
                num_layers = config.get('num_hidden_layers', 0)
                vocab_size = config.get('vocab_size', 0)
                if hidden_size and num_layers and vocab_size:
                    # 简单估算，实际参数量可能与此不同
                    params = hidden_size * hidden_size * num_layers * 4 + hidden_size * vocab_size
                else:
                    return "未知"
            
            # 格式化参数量显示
            if params >= 1e9:
                return f"{params/1e9:.1f}B"
            elif params >= 1e6:
                return f"{params/1e6:.1f}M"
            else:
                return f"{params:,}"
        except Exception:
            return "未知"
    def export_model(self):
        if not self.save_path.get():
            messagebox.showerror("错误", "请选择保存路径")
            return
            
        try:
            export_path = os.path.join(self.save_path.get(), "exported_model")
            os.makedirs(export_path, exist_ok=True)
            
            self.log_text.insert(tk.END, f"正在导出模型到 {export_path}...\n")
            
            # 保存训练配置
            config = {
                "base_model": self.model_var.get(),
                "learning_rate": self.lr_var.get(),
                "batch_size": self.batch_size_var.get(),
                "epochs": self.epochs_var.get(),
                "max_length": self.max_length_var.get(),
                "use_lora": self.use_lora.get(),
                "lora_rank": self.lora_rank.get(),
                "use_fp16": self.use_fp16.get(),
                "gradient_accumulation_steps": self.grad_accum.get()
            }
            
            with open(os.path.join(export_path, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 实际的模型保存代码
            from transformers import AutoTokenizer
            from unsloth import FastLanguageModel
            import torch
            from peft import get_peft_model, LoraConfig
            
            # 加载最新的模型
            model_name = self.model_var.get()
            max_length = int(self.max_length_var.get())
            
            # 加载tokenizer
            self.log_text.insert(tk.END, "正在加载tokenizer...\n")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 加载最新保存的模型
            latest_epoch = int(self.epochs_var.get())
            latest_model_path = os.path.join(self.save_path.get(), f"epoch_{latest_epoch}")
            
            if os.path.exists(latest_model_path):
                self.log_text.insert(tk.END, f"正在加载最新模型: {latest_model_path}...\n")
                
                # 加载模型
                if self.use_lora.get():
                    # 加载基础模型
                    base_model = FastLanguageModel.from_pretrained(
                        model_name=model_name,
                        max_seq_length=max_length,
                        dtype=torch.float16 if self.use_fp16.get() else torch.float32
                    )
                    
                    # 配置LoRA
                    lora_config = LoraConfig(
                        r=int(self.lora_rank.get()),
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                    
                    # 加载LoRA权重
                    model = get_peft_model(base_model, lora_config)
                    model.load_adapter(latest_model_path)
                else:
                    # 直接加载完整模型
                    model = FastLanguageModel.from_pretrained(
                        model_name=latest_model_path,
                        max_seq_length=max_length,
                        dtype=torch.float16 if self.use_fp16.get() else torch.float32
                    )
                
                # 保存模型到导出目录
                self.log_text.insert(tk.END, "正在保存模型...\n")
                model.save_pretrained(export_path)
                tokenizer.save_pretrained(export_path)
                
                self.log_text.insert(tk.END, "模型导出完成！\n")
                messagebox.showinfo("成功", "模型导出完成！")
            else:
                raise FileNotFoundError(f"找不到训练好的模型: {latest_model_path}")
            
        except Exception as e:
            self.log_text.insert(tk.END, f"导出失败: {str(e)}\n")
            messagebox.showerror("错误", f"导出失败: {str(e)}")
    def training_process(self):
        try:
            self.log_text.insert(tk.END, "正在初始化训练...\n")
            
            # 验证输入参数
            if not self.data_path.get():
                raise ValueError("请选择训练数据文件")
            if not os.path.exists(self.data_path.get()):
                raise ValueError("训练数据文件不存在")
                
            # 创建保存目录
            os.makedirs(self.save_path.get(), exist_ok=True)
            
            # 加载训练数据
            with open(self.data_path.get(), 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # 设置网络参数
            if self.proxy.get():
                os.environ["HTTP_PROXY"] = self.proxy.get()
                os.environ["HTTPS_PROXY"] = self.proxy.get()
                self.log_text.insert(tk.END, f"已设置代理: {self.proxy.get()}\n")
            
            # 设置请求超时和重试
            try:
                timeout = int(self.timeout.get())
                retries = int(self.max_retries.get())
                requests.adapters.DEFAULT_RETRIES = retries
                requests.DEFAULT_TIMEOUT = timeout
                self.log_text.insert(tk.END, f"已设置连接超时: {timeout}秒, 重试次数: {retries}\n")
            except ValueError:
                self.log_text.insert(tk.END, "超时或重试次数设置无效，使用默认值\n")
            
            # 导入必要的库
            from transformers import AutoTokenizer, TrainingArguments
            from unsloth import FastLanguageModel, is_bfloat16_supported
            from trl import SFTTrainer
            import torch
            from datasets import Dataset
            import bitsandbytes as bnb
            
            # 获取训练参数
            model_name = self.model_var.get()
            learning_rate = float(self.lr_var.get())
            batch_size = int(self.batch_size_var.get())
            epochs = int(self.epochs_var.get())
            max_length = int(self.max_length_var.get())
            grad_accum_steps = int(self.grad_accum.get())
            
            # 初始化模型和tokenizer
            self.log_text.insert(tk.END, "正在加载模型和tokenizer...\n")
            
            # 添加错误处理和重试机制
            max_attempts = retries
            attempt = 0
            success = False
            
            # 设置量化参数
            load_kwargs = {
                "model_name": model_name,
                "max_seq_length": max_length,
            }
            
            if self.use_4bit.get():
                load_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.float16,
                })
                self.log_text.insert(tk.END, "启用4-bit量化训练...\n")
            elif self.use_8bit.get():
                load_kwargs.update({"load_in_8bit": True})
                self.log_text.insert(tk.END, "启用8-bit量化训练...\n")
            else:
                load_kwargs.update({"dtype": torch.float16 if self.use_fp16.get() else torch.float32})
            
            while attempt < max_attempts and not success:
                try:
                    attempt += 1
                    if self.offline_mode.get():
                        self.log_text.insert(tk.END, "使用离线模式加载本地模型...\n")
                        local_model_path = os.path.join(self.local_model_dir.get(), os.path.basename(model_name))
                        if os.path.exists(local_model_path):
                            load_kwargs["model_name"] = local_model_path
                            model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                        else:
                            self.log_text.insert(tk.END, f"错误: 本地模型路径 {local_model_path} 不存在\n")
                            self.log_text.insert(tk.END, "请确保模型已下载到本地模型目录，或取消勾选离线模式\n")
                            raise FileNotFoundError(f"本地模型不存在: {local_model_path}")
                    else:
                        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                    success = True
                except Exception as e:
                    if attempt < max_attempts:
                        wait_time = attempt * 5
                        self.log_text.insert(tk.END, f"加载失败 (尝试 {attempt}/{max_attempts}): {str(e)}\n")
                        self.log_text.insert(tk.END, f"等待 {wait_time} 秒后重试...\n")
                        self.root.update()
                        time.sleep(wait_time)
                    else:
                        self.log_text.insert(tk.END, f"加载失败，已达到最大重试次数: {str(e)}\n")
                        raise
            
            if not success:
                raise Exception("模型加载失败，请检查网络连接或使用离线模式")
                
            self.log_text.insert(tk.END, "模型加载成功!\n")
            
            # 配置LoRA
            if self.use_lora.get():
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=int(self.lora_rank.get()),
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.log_text.insert(tk.END, "已启用LoRA配置\n")
            
            # 准备数据集
            dataset = Dataset.from_list([{"text": item["text"]} for item in training_data])
            
            # 配置训练参数
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                learning_rate=learning_rate,
                num_train_epochs=epochs,
                fp16=not is_bfloat16_supported() and not (self.use_4bit.get() or self.use_8bit.get()),
                bf16=is_bfloat16_supported() and not (self.use_4bit.get() or self.use_8bit.get()),
                logging_steps=1,
                optim=self.optimizer_var.get(),
                weight_decay=float(self.weight_decay.get()),
                lr_scheduler_type=self.lr_scheduler.get(),
                output_dir=self.save_path.get(),
                save_strategy="steps",
                save_steps=100,
                report_to="none",
                remove_unused_columns=False,
            )
            
            # 初始化SFTTrainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=max_length,
                dataset_num_proc=int(self.num_proc.get()),
                packing=self.use_packing.get(),
                args=training_args,
            )
            
            # 训练循环
            try:
                self.best_loss = float('inf')
                patience = 3  # 早停耐心值
                no_improve = 0  # 未改善次数
                
                for epoch in range(epochs):
                    if not self.training_active:
                        raise InterruptedError("训练被用户中断")
                    
                    self.current_epoch = epoch + 1
                    self.log_text.insert(tk.END, f"\n开始训练 Epoch {self.current_epoch}/{epochs}\n")
                    
                    # 训练一个epoch
                    train_results = trainer.train()
                    
                    # 更新进度条和可视化
                    progress = (self.current_epoch / epochs) * 100
                    self.progress['value'] = progress
                    
                    # 更新日志和可视化
                    loss = train_results.training_loss
                    self.log_text.insert(tk.END, f"Epoch {self.current_epoch}/{epochs}, Loss: {loss:.4f}\n")
                    self.log_text.see(tk.END)
                    self.update_visualization(loss, learning_rate)
                    self.root.update()
                    
                    # 早停检查
                    if loss < self.best_loss:
                        self.best_loss = loss
                        no_improve = 0
                        # 保存最佳模型
                        best_model_path = os.path.join(self.save_path.get(), "best_model")
                        trainer.save_model(best_model_path)
                        tokenizer.save_pretrained(os.path.join(best_model_path, "tokenizer"))
                        self.log_text.insert(tk.END, f"发现更好的模型，已保存到 {best_model_path}\n")
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            self.log_text.insert(tk.END, f"训练loss连续{patience}个epoch未改善，触发早停机制\n")
                            break
                    
                    # 保存当前epoch的模型
                    save_path = os.path.join(self.save_path.get(), f"epoch_{self.current_epoch}")
                    trainer.save_model(save_path)
                    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
                    
                    self.log_text.insert(tk.END, f"Epoch {self.current_epoch} 完成，模型已保存到 {save_path}\n")
                    self.log_text.see(tk.END)
                
                self.log_text.insert(tk.END, "\n训练完成！\n")
                self.log_text.insert(tk.END, f"最佳loss: {self.best_loss:.4f}\n")
                self.progress['value'] = 100
                
            except Exception as e:
                if not isinstance(e, InterruptedError):
                    raise e
            
        except Exception as e:
            self.log_text.insert(tk.END, f"训练出错: {str(e)}\n")
            messagebox.showerror("错误", f"训练出错: {str(e)}")
            raise
        finally:
            self.training_active = False
    def update_model_options(self, event=None):
        """根据选择的模型系列更新模型大小选项"""
        family = self.model_family.get()
        if family in self.model_mapping:
            sizes = list(self.model_mapping[family].keys())
            self.model_size_combo["values"] = sizes
            if sizes:
                self.model_size.set(sizes[0])
                self.update_model_name()
        else:
            self.model_size_combo["values"] = ()
            self.model_size.set("")
    def select_model_dir(self):
        """选择本地模型目录"""
        directory = filedialog.askdirectory(title="选择本地模型目录")
        if directory:
            self.local_model_dir.set(directory)
            self.log_text.insert(tk.END, f"已设置本地模型目录: {directory}\n")
    
    def update_model_name(self, event=None):
        """根据选择的模型系列和大小更新完整模型名称"""
        family = self.model_family.get()
        size = self.model_size.get()
        
        if family in self.model_mapping and size in self.model_mapping[family]:
            model_name = self.model_mapping[family][size]
            self.model_var.set(model_name)

if __name__ == "__main__":
    root = tk.Tk()
    app = FineTuningGUI(root)
    root.mainloop()