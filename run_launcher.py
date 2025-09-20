# run_launcher.py (升级版 v2.0)
import os
import sys
import time
import re # 导入正则表达式库用于解析文件名

# --- 初始化环境 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my_utils.init_utils import seed_torch, seed_tensorflow

# --- 导入各个模型的学习器类 ---
from Models.ProtoNet.proto_train import ProtoNet_learner
from Models.MAML.maml_train import MAML_learner
from Models.MAML.reptile_train import Reptile_learner
from Models.RelationNet.relation_train import RelationNet_learner
from Models.CNN_torch.cnn_ft_train import CNN_FT_learner
from Models.CNN_torch.cnn_mmd_train import CNN_MMD_learner
from Models.CNN.cnn_train import CNN_learner as CNN_TF_learner

# --- 全局路径配置 ---
WEIGHTS_DIR = "Models weights"
SAVE_DIR = "Models_trained_by_me"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 辅助函数 ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_choice(options):
    while True:
        try:
            choice = int(input("\n>>> 请输入您的选择 (数字): "))
            if choice in options:
                return choice
            else:
                print("无效选择，请输入列表中的数字。")
        except ValueError:
            print("无效输入，请输入一个数字。")

def press_enter_to_continue():
    input("\n按 Enter 键返回菜单...")

# === 新增功能：动态构建测试选项 ===
def build_test_options_for_model(model_name, original_weights):
    """动态扫描文件夹，为单个模型构建测试选项"""
    all_weights = {}
    
    # 1. 添加原始的预训练模型
    for key, info in original_weights.items():
        # (文件名, ways, shots, *其他参数, 所在文件夹)
        all_weights[key] = info + (WEIGHTS_DIR,)

    # 2. 扫描并添加用户自己训练的模型
    current_key = len(all_weights) + 1
    if os.path.exists(SAVE_DIR):
        for filename in sorted(os.listdir(SAVE_DIR)):
            if filename.lower().startswith(model_name.lower()):
                # 尝试从文件名解析 ways 和 shots
                ways_match = re.search(r'ways(\d+)', filename)
                shots_match = re.search(r'shots(\d+)', filename)
                ways = int(ways_match.group(1)) if ways_match else 0
                shots = int(shots_match.group(1)) if shots_match else 0
                
                # 创建新的描述信息
                new_info = (filename, ways, shots)
                
                # 检查是否有额外参数 (例如 Reptile)
                if model_name == "Reptile":
                    new_info += (30,) # 默认 inner_steps
                
                all_weights[current_key] = new_info + (SAVE_DIR,)
                current_key += 1
                
    return all_weights

# === 修改后的测试逻辑 ===
def run_test_logic(model_name, chosen_info):
    """根据选择的信息来运行测试"""
    # 解包选项信息
    filename, ways, shots = chosen_info[0], chosen_info[1], chosen_info[2]
    folder_path = chosen_info[-1]
    load_path = os.path.join(folder_path, filename)

    clear_screen()
    print(f"--- 正在测试模型: {filename} ---")
    print(f"来源文件夹: {folder_path}")
    print(f"参数: ways={ways}, shots={shots}")
    print("-" * 30)
    
    # 确保 ways 和 shots > 0
    if ways == 0 or shots == 0:
        print("\n[错误] 无法从文件名中解析出 ways 和 shots，无法进行测试。")
        print("请确保您训练的模型文件名包含 'waysX' 和 'shotsY' (例如: ProtoNet_ways10_shots5_...)")
        press_enter_to_continue()
        return

    print(f"\n[INFO] 正在初始化 {model_name} (ways={ways})...")
    seed_torch(2021)
    
    if model_name == "ProtoNet":
        net = ProtoNet_learner(ways=ways)
        net.test(load_path, shots=shots)
    elif model_name == "CNN-FT":
        net = CNN_FT_learner(ways=ways)
        net.test_cnn_ft(load_path, shots=shots)
    elif model_name == "CNN-MMD":
        net = CNN_MMD_learner(ways=ways, shots=shots)
        net.test_cnn(load_path)
    elif model_name == "MAML":
        net = MAML_learner(ways=ways)
        net.test(load_path, shots=shots)
    elif model_name == "RelationNet":
        net = RelationNet_learner(ways=ways)
        net.test(load_path, shots=shots)
    elif model_name == "Reptile":
        inner_steps = chosen_info[3]
        net = Reptile_learner(ways=ways)
        net.test(load_path, shots=shots, inner_test_steps=inner_steps)

    press_enter_to_continue()


def test_menu():
    original_weights_map = {
        "ProtoNet": {
            1: ("ProtoNet_C30_ep50", 10, 5),
            2: ("ProtoNet_T2_ep62", 4, 5)
        },
        "CNN-FT": {
            1: ("cnn_ft_C30_ep50", 10, 5),
            2: ("cnn_ft_C30_ep72", 10, 5)
        },
        "CNN-MMD": {
            1: ("cnn_mmd_C30_ep50", 10, 5)
        },
        "MAML": {
            1: ("MAML_C30_ep457", 10, 5),
            2: ("MAML_T2_ep414", 4, 5)
        },
        "RelationNet": {
            1: ("RelationNet_C30_ep200", 10, 5),
            2: ("RelationNet_C30_ep252", 10, 5),
            3: ("RelationNet_T2_ep284", 4, 5),
            4: ("RelationNet_T2_ep394", 4, 5)
        },
        "Reptile": {
            1: ("Reptile_C30_ep730", 10, 5, 30),
            2: ("Reptile_T2_ep702", 4, 1, 30)
        }
    }

    test_main_menu_options = {
        1: "ProtoNet", 2: "CNN-FT", 3: "CNN-MMD", 4: "MAML", 5: "RelationNet", 6: "Reptile", 7: "返回主菜单"
    }

    while True:
        clear_screen()
        print("====== 测试模型：请选择模型类型 ======")
        for i, name in test_main_menu_options.items():
            print(f"{i}. {name}")

        choice = get_choice(test_main_menu_options)

        if choice == 7:
            break
        
        model_name = test_main_menu_options[choice]
        original_weights = original_weights_map[model_name]
        
        # 动态构建选项
        dynamic_options = build_test_options_for_model(model_name, original_weights)
        
        # 显示动态菜单
        clear_screen()
        print(f"--- {model_name} 模型权重选择 ---")
        print(f"扫描文件夹: '{WEIGHTS_DIR}' 和 '{SAVE_DIR}'")
        for i, info in dynamic_options.items():
            folder_tag = " [原始]" if info[-1] == WEIGHTS_DIR else " [新训练]"
            print(f"{i}. {info[0]} (ways={info[1]}, shots={info[2]}){folder_tag}")
        print(f"{len(dynamic_options) + 1}. 返回上一级")

        sub_choice_options = list(dynamic_options.keys()) + [len(dynamic_options) + 1]
        sub_choice = get_choice(sub_choice_options)

        if sub_choice == len(dynamic_options) + 1:
            continue
            
        chosen_info = dynamic_options[sub_choice]
        run_test_logic(model_name, chosen_info)

# --- 训练功能模块 (保持不变) ---
def run_train_logic(model_name):
    clear_screen()
    print(f"--- 训练新模型: {model_name} ---")
    try:
        ways = int(input("请输入分类数 (ways, 例如 10 或 4): "))
        shots = int(input("请输入样本数 (shots, 例如 5 或 1): "))
    except ValueError:
        print("输入无效，请输入数字。")
        press_enter_to_continue()
        return

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # 确保文件名包含 ways 和 shots 信息
    model_savename = f"{model_name}_ways{ways}_shots{shots}_{timestr}.pth"
    save_path = os.path.join(SAVE_DIR, model_savename)
    
    print(f"\n[INFO] 正在初始化 {model_name} (ways={ways}, shots={shots})...")
    print(f"[INFO] 模型将保存在: {save_path} (训练结束后)")
    
    if model_name in ["ProtoNet", "MAML", "RelationNet", "Reptile", "CNN-FT", "CNN-MMD"]:
        seed_torch(2021)
        if model_name == "ProtoNet":
            net = ProtoNet_learner(ways=ways)
            net.train(save_path, shots=shots)
        # ... (其他模型的训练调用保持不变)
        elif model_name == "CNN-FT":
            net = CNN_FT_learner(ways=ways)
            net.train_cnn(save_path, shots=shots)
        elif model_name == "CNN-MMD":
            net = CNN_MMD_learner(ways=ways, shots=shots)
            net.train_cnn(save_path)
        elif model_name == "MAML":
            net = MAML_learner(ways=ways)
            net.train(save_path, shots=shots)
        elif model_name == "RelationNet":
            net = RelationNet_learner(ways=ways)
            net.train(save_path, shots=shots)
        elif model_name == "Reptile":
            net = Reptile_learner(ways=ways)
            net.train(save_path, shots=shots)
            
    elif model_name == "CNN (TensorFlow)":
        seed_tensorflow(2021)
        print("[WARN] TensorFlow版本的CNN分类数固定为10。")
        net = CNN_TF_learner(num_classes=10)
        net.train(save_path)

    press_enter_to_continue()

def train_menu():
    train_options = {
        1: "ProtoNet", 2: "CNN-FT", 3: "CNN-MMD", 4: "MAML", 5: "RelationNet", 6: "Reptile", 7: "CNN (TensorFlow)", 8: "返回主菜单"
    }
    while True:
        clear_screen()
        print("====== 训练新模型 ======")
        for i, name in train_options.items():
            print(f"{i}. {name}")
        
        choice = get_choice(train_options)
        
        if choice == 8:
            break
            
        model_name = train_options[choice]
        run_train_logic(model_name)

# --- 主菜单 (保持不变) ---
def main_menu():
    main_options = {1: ("测试预训练模型", test_menu), 2: ("训练新模型", train_menu), 3: ("退出", None)}
    while True:
        clear_screen()
        print("="*15, "模型启动器 (v2.0)", "="*15)
        print("项目根目录:", os.getcwd())
        print("预训练模型目录:", os.path.abspath(WEIGHTS_DIR))
        print("新模型保存目录:", os.path.abspath(SAVE_DIR))
        print("="*46)
        for i, (name, _) in main_options.items():
            print(f"{i}. {name}")
        
        choice = get_choice(main_options)
        
        if choice == 3:
            print("程序已退出。")
            break
            
        _, menu_function = main_options[choice]
        menu_function()

if __name__ == "__main__":
    main_menu()