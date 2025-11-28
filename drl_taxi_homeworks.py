import os
import random
import csv

import numpy as np
import matplotlib.pyplot as plt

# gym veya gymnasium
try:
    import gym
except ImportError:  # gym yoksa gymnasium'u gym gibi kullan
    import gymnasium as gym

# GIF için imageio (opsiyonel)
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# .h5 için h5py (opsiyonel)
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# ======================================================================
# GENEL AYARLAR
# ======================================================================

BASE_DIR = r"C:\Users\MSI\Desktop\DRL_TAXI_HOMEWORK"

ALPHA = 0.1          # learning rate
GAMMA = 1.0          # discount factor
EPSILON = 0.1        # exploration rate (sabit)
NUM_EPISODES = 10000 # eğitim epizodu
MAX_STEPS = 200      # epizot başına max adım

TEST_EPISODES = 100  # testte 100 epizot

# GIF için sampling ayarları
TRAIN_GIF_INTERVAL = 100       # her 100. epizot için GIF’e kare al
TRAIN_GIF_MAX_STEPS = 30       # seçilen epizotlarda en fazla 30 adım
TEST_GIF_MAX_STEPS = 30        # her test epizodundan en fazla 30 adım

GIF_DPI = 80                   # çözünürlük biraz düşük, dosya boyutu küçük olsun


# ======================================================================
# ORTAM YARDIMCILARI
# ======================================================================

def reset_env(env):
    """Hem gym hem gymnasium reset çıktısını normalize et."""
    out = env.reset()
    if isinstance(out, tuple):   # (obs, info)
        state = out[0]
    else:
        state = out
    return state


def step_env(env, action):
    """Hem gym hem gymnasium step çıktısını normalize et."""
    out = env.step(action)
    if len(out) == 4:
        next_state, reward, done, info = out
    elif len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise ValueError("Env.step() beklenmeyen uzunlukta çıktı verdi.")
    return next_state, reward, done


def render_base_frame(env):
    """Env'den RGB frame al (Taxi-v3 grid'i)."""
    frame = None
    try:
        frame = env.render()
    except TypeError:
        frame = env.render(mode="rgb_array")
    return frame


def render_titled_frame(env, episode, step, reward, total_reward,
                        total_episodes, mode="Train"):
    """
    Env görüntüsünün üstüne başlık (episode, step, reward) yazar
    ve numpy array olarak döndürür. GIF için kullanacağız.
    """
    base = render_base_frame(env)
    if base is None:
        return None

    fig, ax = plt.subplots(figsize=(3, 3), dpi=GIF_DPI)
    ax.imshow(base)
    ax.axis("off")

    title = (f"{mode} Episode {episode}/{total_episodes} | Step {step}\n"
             f"Reward={reward:.2f} | Total={total_reward:.2f}")
    ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.1)
    fig.canvas.draw()

    # Farklı matplotlib sürümlerine uyumlu dönüşüm
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3]

    plt.close(fig)
    return img


# ======================================================================
# Q-TABLE .H5 KAYDET / YÜKLE
# ======================================================================

def save_q_table_h5(q_table, path_h5):
    """Q-tablosunu .h5 (varsa) yoksa .npy olarak kaydet."""
    if HAS_H5PY:
        import h5py  # içerden de import et
        with h5py.File(path_h5, "w") as f:
            f.create_dataset("q_table", data=q_table)
        print(">> Q-tablosu .h5 olarak kaydedildi:")
        print("   ", path_h5)
    else:
        npy_path = path_h5.replace(".h5", ".npy")
        np.save(npy_path, q_table)
        print(">> Uyarı: h5py kurulu değil, Q-tablosu .npy olarak kaydedildi:")
        print("   ", npy_path)


def load_q_table_h5(path_h5):
    """Q-tablosunu .h5 (veya fallback .npy) dosyasından yükle."""
    if os.path.exists(path_h5) and HAS_H5PY:
        import h5py
        with h5py.File(path_h5, "r") as f:
            return f["q_table"][:]

    npy_path = path_h5.replace(".h5", ".npy")
    if os.path.exists(npy_path):
        print(">> h5 bulunamadı ama .npy bulundu, oradan yükleniyor:")
        print("   ", npy_path)
        return np.load(npy_path)

    raise FileNotFoundError("Ne .h5 ne .npy Q-table dosyası bulunamadı.")


# ======================================================================
# Q-LEARNING EĞİTİMİ
# ======================================================================

def train_q_learning(env,
                     num_episodes=NUM_EPISODES,
                     max_steps=MAX_STEPS,
                     alpha=ALPHA,
                     gamma=GAMMA,
                     epsilon=EPSILON,
                     record_train_gif=True,
                     train_gif_path=None):
    """
    Taxi-v3 ortamında tabular Q-learning eğitimi.

    GIF için:
    - imageio varsa writer açılır,
    - her 100. epizotta, en fazla 30 adım boyunca kareler eklenir.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    episode_rewards = np.zeros(num_episodes, dtype=np.float32)
    total_epochs = np.zeros(num_episodes, dtype=np.int32)

    gif_writer = None
    if record_train_gif and train_gif_path is not None and HAS_IMAGEIO:
        gif_writer = imageio.get_writer(train_gif_path, mode="I", duration=0.2)
        print(">> Eğitim GIF writer açıldı:", train_gif_path)
    elif record_train_gif and not HAS_IMAGEIO:
        print(">> Uyarı: imageio yok, eğitim GIF'i oluşturulmayacak.")

    for episode in range(1, num_episodes + 1):
        state = reset_env(env)   # her epizotta T/P/D rastgele
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Epsilon-greedy politika
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done = step_env(env, action)

            # Q-learning güncellemesi
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (
                reward + gamma * next_max - old_value
            )

            state = next_state
            total_reward += reward
            steps += 1

            # Eğitim GIF'i: her 100. epizot, ilk 30 adım
            if (gif_writer is not None and
                episode % TRAIN_GIF_INTERVAL == 0 and
                steps <= TRAIN_GIF_MAX_STEPS):
                frame = render_titled_frame(
                    env,
                    episode,
                    steps,
                    reward,
                    total_reward,
                    num_episodes,
                    mode="Train",
                )
                if frame is not None:
                    gif_writer.append_data(frame)

        episode_rewards[episode - 1] = total_reward
        total_epochs[episode - 1] = steps

        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Toplam ödül: {total_reward:.1f} | Adım: {steps}")

    print("=== Eğitim tamamlandı ===")

    if gif_writer is not None:
        gif_writer.close()
        print(">> Eğitim GIF'i kaydedildi.")

    return q_table, episode_rewards, total_epochs


# ======================================================================
# TEST İÇİN FARKLI BAŞLANGIÇ DURUMLARI
# ======================================================================

def unique_test_start_state(env, used_starts):
    """
    Testte her epizot için farklı bir başlangıç durumu seç.
    TaxiEnv.decode(state) ile (taxi_row, taxi_col, passenger_idx, dest_idx)
    alıp set'te tutuyoruz.
    """
    while True:
        state = reset_env(env)
        try:
            taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)
        except AttributeError:
            taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)

        sig = (taxi_row, taxi_col, pass_idx, dest_idx)
        if sig not in used_starts:
            used_starts.add(sig)
            return state


# ======================================================================
# TEST POLİTİKA + TEST GIF
# ======================================================================

def test_policy(env,
                q_table,
                num_episodes=TEST_EPISODES,
                max_steps=MAX_STEPS,
                record_test_gif=True,
                test_gif_path=None):
    """
    Eğitilmiş Q-tablosu ile greedy politika testi.
    100 farklı başlangıç durumu kullanılıyor.
    GIF için tüm epizotlarda ilk 30 adım kaydediliyor.
    """
    total_rewards = []
    total_steps = []
    used_starts = set()

    gif_writer = None
    if record_test_gif and test_gif_path is not None and HAS_IMAGEIO:
        gif_writer = imageio.get_writer(test_gif_path, mode="I", duration=0.2)
        print(">> Test GIF writer açıldı:", test_gif_path)
    elif record_test_gif and not HAS_IMAGEIO:
        print(">> Uyarı: imageio yok, test GIF'i oluşturulmayacak.")

    for episode in range(1, num_episodes + 1):
        state = unique_test_start_state(env, used_starts)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = int(np.argmax(q_table[state]))
            next_state, reward, done = step_env(env, action)
            ep_reward += reward
            steps += 1

            # Test GIF'i: her epizotta ilk 30 adım
            if gif_writer is not None and steps <= TEST_GIF_MAX_STEPS:
                frame = render_titled_frame(
                    env,
                    episode,
                    steps,
                    reward,
                    ep_reward,
                    num_episodes,
                    mode="Test",
                )
                if frame is not None:
                    gif_writer.append_data(frame)

            state = next_state

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        print(f"[TEST] Episode {episode}/{num_episodes} | "
              f"Toplam ödül: {ep_reward:.1f} | Adım: {steps}")

    if gif_writer is not None:
        gif_writer.close()
        print(">> Test GIF'i kaydedildi.")

    print("\n=== Test özeti ===")
    print(f"Ortalama ödül: {np.mean(total_rewards):.2f}")
    print(f"Ortalama adım: {np.mean(total_steps):.2f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # Taxi-v3 ortamı
    try:
        env = gym.make("Taxi-v3", render_mode="rgb_array").env
    except TypeError:
        env = gym.make("Taxi-v3").env

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    q_table_path = os.path.join(BASE_DIR, "taxi_q_table.h5")
    results_csv_path = os.path.join(BASE_DIR, "training_results.csv")
    reward_plot_path = os.path.join(BASE_DIR, "training_rewards_plot.png")
    train_gif_path = os.path.join(BASE_DIR, "taxi_train_samples.gif")
    test_gif_path = os.path.join(BASE_DIR, "taxi_test_100_episodes.gif")

    # 1) Daha önce eğitilmiş Q-table varsa dosyadan yükle
    if os.path.exists(q_table_path) or os.path.exists(q_table_path.replace(".h5", ".npy")):
        print("\n>> Kayıtlı Q-table bulundu, dosyadan yükleniyor...")
        q_table = load_q_table_h5(q_table_path)
        episode_rewards = None
        total_epochs = None

    # 2) Yoksa Q-learning ile eğit
    else:
        print("\n>> Kayıtlı Q-table yok, eğitim başlıyor...")
        q_table, episode_rewards, total_epochs = train_q_learning(
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON,
            record_train_gif=True,
            train_gif_path=train_gif_path,
        )

        # Q-table'ı kaydet
        save_q_table_h5(q_table, q_table_path)

        # Episode başına ödül ve adım sayısını CSV'ye yaz
        if episode_rewards is not None and total_epochs is not None:
            with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "steps"])
                for i, (r, s) in enumerate(zip(episode_rewards, total_epochs), start=1):
                    writer.writerow([i, float(r), int(s)])
            print(">> Eğitim sonuçları CSV olarak kaydedildi:")
            print("   ", results_csv_path)

            # Reward–episode grafiği (hem göster, hem PNG kaydet)
            plt.figure()
            plt.title("Total reward per episode")
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.plot(episode_rewards)
            plt.tight_layout()
            plt.savefig(reward_plot_path, dpi=150)
            print(">> Reward–episode grafiği kaydedildi:")
            print("   ", reward_plot_path)
            plt.show()

    # 3) Eğitilmiş ajanla test ve test GIF'i
    print("\n>> Eğitilmiş ajan ile test başlıyor...")
    test_policy(
        env,
        q_table,
        num_episodes=TEST_EPISODES,
        max_steps=MAX_STEPS,
        record_test_gif=True,
        test_gif_path=test_gif_path,
    )


if __name__ == "__main__":
    main()
