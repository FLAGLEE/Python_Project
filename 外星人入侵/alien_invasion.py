import pygame
from pygame.sprite import Group

from settings import Settings
from game_stats import GameStats
from ship import Ship
from button import Button
from scoreboard import Scoreboard
import game_functions as gf


def run_game():
    # 初始化游戏并创建一个屏幕对象
    pygame.init()
    ai_settings = Settings()

    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption("Alien Invasion")

    # 创建Play按钮
    play_button = Button(ai_settings, screen, "Play")

    # 创建存储游戏统计信息的实例， 并创建记分牌
    stats = GameStats(ai_settings)
    sb = Scoreboard(ai_settings, screen, stats)

    # 创建一艘飞船
    ship = Ship(ai_settings, screen)
    # 创建一个用于存储子弹的编组
    bullets = Group()
    # 创建一个外星人
    # alien = Alien(ai_settings, screen)
    # 创建一个外星人编组
    aliens = Group()

    # 创建外星人群
    gf.create_fleet(ai_settings, screen, ship, aliens)

    # 开始游戏主循环
    while True:
        # 监视键盘和鼠标事件
        gf.check_event(ai_settings, stats, play_button, screen, sb, ship, aliens, bullets)
        if stats.game_active:
            # 更新飞船状态
            ship.update()
            # 更新子弹的位置， 并删除已消失的子弹
            gf.update_bullets(ai_settings, screen, stats, sb, ship, aliens, bullets)
            # 更新外星人位置
            gf.update_aliens(ai_settings, stats, screen, sb, ship, aliens, bullets)
        # 每次循环时都重绘屏幕
        gf.update_screen(ai_settings, screen, stats, sb, ship, aliens, bullets, play_button)


if __name__ == '__main__':
    run_game()
