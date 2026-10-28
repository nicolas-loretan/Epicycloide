"""
=============================================================================
 EPICYCLOIDE  -  Dessiner n'importe quelle courbe avec des epicycles (Fourier)
=============================================================================
 Code source complet et annote (annexe du travail de maturite).

 CE QUE FAIT LE PROGRAMME
 L'utilisateur dessine a la souris. Le programme transforme ce dessin en une
 seule courbe fermee, calcule sa serie de Fourier discrete, puis rejoue le
 dessin avec une chaine de cercles tournants mis bout a bout : le bout du
 dernier cercle retrace la courbe.

 POURQUOI TOUTES CES ETAPES
 La theorie de Fourier exige trois choses qu'un dessin a la souris ne fournit
 jamais : une courbe UNIQUE (or on leve le crayon), FERMEE (or un trait a deux
 bouts distincts) et parcourue a VITESSE CONSTANTE (or la souris va vite ou
 lentement). Les classes GeometryEngine et FourierEngine ne servent qu'a
 corriger ces trois defauts avant le calcul proprement dit.

 Bibliotheques externes :
   - pygame : creation de la fenetre, dessin a l'ecran, gestion souris/clavier.
   - numpy  : calcul sur de grands tableaux de nombres (interpolation, DFT).

 Organisation du fichier (de haut en bas) :
   1. Constantes globales
   2. Modele de donnees      : ScreenName, AppState, Layout
   3. Widgets               : ValueBox, Button
   4. Ecran de base         : BaseScreen
   5. Moteurs de calcul     : FourierEngine, GeometryEngine   (sans pygame)
   6. Ecrans concrets       : DrawScreen, AnimateScreen
   7. Application + boucle   : App
=============================================================================
"""

import heapq          # file de priorite : rend toujours le plus petit element. Utilisee par Dijkstra.
import math           # cos, sin, hypot (longueur d'un vecteur), inf (l'infini)
from collections import defaultdict   # dictionnaire qui cree tout seul une valeur par defaut
from dataclasses import dataclass, field   # fabrique automatiquement le constructeur d'une classe
from enum import Enum, auto            # permet de definir des constantes nommees
from typing import Callable, Optional  # annotations de type : indications pour le lecteur, sans effet
import numpy as np    # convention universelle : on importe numpy sous le nom np
import pygame


# =========================
# Configuration generale
# =========================

WIDTH, HEIGHT = 1000, 666        # taille initiale de la fenetre, en pixels
FPS = 124                        # nombre d'images par seconde de la boucle principale
DRAWING_ZONE_RATIO = 1.5         # rapport largeur/hauteur impose a la zone de dessin
NB_SAMPLES = 1024                # nombre de points echantillonnes le long du trace avant la DFT
PROGRESS_BAR_H = 12              # hauteur de la barre de progression, en pixels

# Reglage de vitesse de l'animation. La boite de reglage affichee a l'ecran
# manipule un ENTIER, plus lisible ; la vitesse reellement utilisee dans les
# calculs vaut cet entier divise par SPEED_DIVISOR.
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 1, 1000, 1
SPEED_DIVISOR = 10.0

pygame.init()                    # demarre tous les modules de pygame ; obligatoire avant tout le reste
# set_repeat fait qu'une touche maintenue enfoncee produit des evenements repetes :
# 300 ms d'attente avant le premier, puis un toutes les 50 ms. C'est ce qui permet
# a la boite de reglage de faire defiler la vitesse quand on garde la fleche appuyee.
pygame.key.set_repeat(300, 50)


# =========================
# Modele / etat partage
# =========================

class ScreenName(Enum):
    """Les identifiants des deux ecrans du programme, plus le signal de fermeture.

    Un Enum est un ensemble de constantes nommees. auto() attribue a chacune une
    valeur entiere automatique et unique, dont on n'a jamais besoin de connaitre
    la valeur : on ecrit ScreenName.DRAW plutot que 0 ou autre si on utilisait des listes par exemples, ce qui rend le code lisible
    et interdit les fautes de frappe silencieuses.

    Sert a indiquer vers quel ecran basculer : un ecran remplit son attribut
    next_screen avec une de ces trois valeurs, et la boucle principale agit."""
    DRAW = auto()       # l'ecran ou l'utilisateur dessine
    ANIMATE = auto()    # l'ecran ou l'animation se joue
    QUIT = auto()       # demande de fermeture du programme


@dataclass
class AppState:
    """Les donnees que les deux ecrans se partagent et qui survivent au changement d'ecran.

    Champs :
      paths        : le dessin lui-meme. C'est une liste de traits, et chaque trait
                     est une liste de points (x, y). Un nouveau trait commence des
                     que l'utilisateur releve le bouton de la souris.
      column_ratio : largeur de la colonne de boutons, exprimee en fraction de la
                     largeur de la fenetre (0.2 = un cinquieme). L'utilisateur peut
                     la modifier en faisant glisser la bordure.
      border_drag  : vaut True pendant que l'utilisateur fait justement glisser
                     cette bordure, pour que le programme sache ignorer le dessin.

    Le decorateur @dataclass, importe du module dataclasses, ecrit tout seul le
    constructeur __init__ a partir de la liste d'attributs ci-dessous. Sans lui, il
    faudrait ecrire a la main 'def __init__(self, paths, column_ratio, ...)' puis
    'self.paths = paths' pour chaque champ. La classe ne sert qu'a transporter des
    donnees, aucune logique n'y figure."""
    paths: list[list[tuple[float, float]]] = field(default_factory=list)
    column_ratio: float = 0.2
    border_drag: bool = False

# Pourquoi field(default_factory=list) et non simplement [] : une valeur par defaut
# qui est une liste serait creee UNE SEULE FOIS, au chargement du fichier, et
# partagee par tous les AppState fabriques ensuite. Deux dessins se melangeraient.
# default_factory=list demande au contraire de fabriquer une liste neuve a chaque
# nouvel objet.


@dataclass
class Layout:
    """Les trois zones rectangulaires de la fenetre, recalculees a chaque redimensionnement.

    Champs :
      rect_col       : la colonne de droite, ou sont empiles les boutons.
      rect_info      : le bandeau du haut, ou sont la boite de reglage de vitesse
                       et la barre de progression.
      rect_main      : la grande zone centrale, ou l'on dessine et ou l'animation
                       se joue.
      header_at_min  : vaut True quand le bandeau du haut a ete comprime jusqu'a sa
                       hauteur minimale. Sert a empecher l'utilisateur de continuer
                       a elargir la colonne de boutons, ce qui ecraserait le bandeau.

    Un pygame.Rect est un objet de pygame decrivant un rectangle par quatre nombres
    (x, y, largeur, hauteur) et fournissant des raccourcis pratiques : .bottom pour
    le bas, .centery pour le milieu vertical, .collidepoint(p) pour tester si un
    point tombe dedans."""
    rect_col: pygame.Rect
    rect_info: pygame.Rect
    rect_main: pygame.Rect
    header_at_min: bool


# =========================
# Interface utilisateur
# =========================

class ValueBox:
    """Une boite affichant un nombre entier que l'utilisateur peut regler au clavier.

    Le programme n'en utilise qu'une seule : le reglage de vitesse de l'animation.
    On clique dessus pour la selectionner, puis les fleches haut et bas modifient la
    valeur. Le pas d'incrementation ACCELERE quand on maintient la touche enfoncee,
    sans quoi il faudrait mille appuis pour parcourir toute la plage de vitesse."""

    # Table du pas d'incrementation, lue de haut en bas : "a partir de N repetitions
    # de la touche, le pas vaut P". Avec la repetition reglee a 50 ms, 30 repetitions
    # correspondent a environ une seconde et demie de maintien.
    ACCELERATION = [(30, 50), (20, 20), (10, 5), (0, 1)]

    def __init__(
        self,
        label: str,
        getter: Callable[[], int],
        setter: Callable[[int], None],
        min_val: int,
        max_val: int,
    ):
        """Prepare une boite de reglage.

        Arguments :
          label   : le texte affiche devant la valeur, par exemple "Vitesse".
          getter  : une fonction sans argument qui, appelee, renvoie la valeur
                    actuelle du reglage. La boite s'en sert pour savoir quoi
                    afficher.
          setter  : une fonction prenant un entier, qui enregistre la nouvelle
                    valeur la ou elle doit aller. La boite s'en sert quand
                    l'utilisateur appuie sur une fleche.
          min_val : valeur en dessous de laquelle on ne peut pas descendre.
          max_val : valeur au-dessus de laquelle on ne peut pas monter.

        Pourquoi un getter et un setter plutot que la valeur elle-meme : la boite ne
        garde ainsi aucune copie du reglage. Elle lit et ecrit directement dans
        l'attribut qui fait autorite, donc l'affichage ne peut jamais se
        desynchroniser de l'etat reel du programme."""
        self.rect = pygame.Rect(0, 0, 0, 0)  # position a l'ecran, recalculee a chaque affichage
        self.label = label
        self.getter = getter
        self.setter = setter
        self.min_val = min_val
        self.max_val = max_val
        self.selected = False                # True quand l'utilisateur a clique dans la boite
        self.held_key: Optional[int] = None  # code de la fleche actuellement maintenue, ou None
        self.hold_count = 0                  # nombre de repetitions recues depuis l'appui

    @property
    def value(self) -> int:
        """Renvoie la valeur actuelle du reglage, en interrogeant le getter.

        @property est un outil integre au langage Python, sans importation. Il
        permet d'ecrire 'boite.value', sans parentheses, exactement comme si on
        lisait une variable, alors qu'un appel de fonction a lieu en coulisse. Cela
        cache le getter derriere une syntaxe simple."""
        return self.getter()

    @value.setter
    def value(self, new_value: int) -> None:
        """Enregistre une nouvelle valeur, en la transmettant au setter.

        Arguments :
          new_value : le nouvel entier a enregistrer.

        @value.setter est le pendant de @property pour l'ecriture : grace a lui,
        l'instruction 'boite.value = 12' declenche en realite cette methode, qui
        appelle le setter fourni au constructeur."""
        self.setter(new_value)

    def _step(self) -> int:
        """Renvoie de combien la valeur doit changer au prochain appui.

        Ne sert qu'a handle_event. Parcourt la table ACCELERATION de haut en bas et
        renvoie le premier pas dont le seuil est atteint : plus la touche est
        maintenue longtemps, plus hold_count est eleve, plus le pas est gros. Le
        souligne au debut du nom signale, par convention, une methode interne a la
        classe, que rien d'exterieur n'a besoin d'appeler."""
        for threshold, step in ValueBox.ACCELERATION:
            if self.hold_count >= threshold:
                return step
        return 1

    def draw(self, surface: pygame.Surface, index: int, total: int,
             container: pygame.Rect, font: pygame.font.Font) -> None:
        """Dessine la boite a l'ecran et memorise sa position pour la detection des clics.

        Arguments :
          surface   : la zone de pixels sur laquelle dessiner, c'est-a-dire la fenetre.
          index     : le rang de cette boite parmi celles de l'ecran, a partir de 0.
          total     : le nombre total de boites a afficher. Avec index, il determine
                      quelle part de la largeur revient a cette boite.
          container : le rectangle dans lequel les boites doivent tenir. Ce n'est pas
                      forcement le bandeau entier : quand une barre de progression est
                      affichee, on passe ici le bandeau ampute de sa hauteur.
          font      : la police avec laquelle ecrire le texte."""
        # Les boites se partagent equitablement la largeur du conteneur.
        box_w = container.width / total
        self.rect = pygame.Rect(index * box_w, container.y, box_w, container.height)

        color = (255, 255, 0) if self.selected else (200, 200, 200)   # jaune si selectionnee
        pygame.draw.rect(surface, (30, 30, 30), self.rect)     # le fond, gris tres fonce
        # Le quatrieme argument de draw.rect est l'epaisseur du trait. Quand il est
        # present, pygame dessine un cadre creux au lieu d'un rectangle plein.
        pygame.draw.rect(surface, color, self.rect, 1)

        # font.render ne dessine rien directement : il fabrique une petite image
        # contenant le texte. Le True au milieu active le lissage des caracteres.
        text = font.render(f"{self.label}: {self.value}", True, color)
        # blit colle cette image sur la surface. get_rect(center=...) calcule la
        # position qui centre l'image sur le point demande.
        surface.blit(text, text.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> None:
        """Reagit a une action de l'utilisateur : clic de selection, ou fleche de reglage.

        Arguments :
          event : un evenement fourni par pygame. Chaque evenement porte un type
                  (clic enfonce, touche relachee, souris deplacee...) et des donnees
                  qui dependent de ce type.

        Cette methode recoit TOUS les evenements de l'ecran, y compris ceux qui ne la
        concernent pas ; elle commence donc par trier sur le type."""
        # Un clic quelque part : la boite devient selectionnee si le clic est tombe
        # dedans, et cesse de l'etre sinon. collidepoint teste l'appartenance d'un
        # point a un rectangle.
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.selected = self.rect.collidepoint(event.pos)

        # Touche relachee : on oublie la touche maintenue, pour que le prochain appui
        # isole reparte d'un pas de 1.
        elif event.type == pygame.KEYUP:
            self.held_key = None
            self.hold_count = 0

        elif self.selected and event.type == pygame.KEYDOWN and event.key in (pygame.K_UP, pygame.K_DOWN):
            # Si la touche est la meme que la fois precedente, c'est que pygame
            # repete un appui maintenu : on incremente le compteur pour accelerer.
            # Sinon, c'est un nouvel appui et on repart de zero.
            if event.key == self.held_key:
                self.hold_count += 1
            else:
                self.held_key = event.key
                self.hold_count = 0

            step = self._step()
            # min et max servent ici a rester dans les bornes autorisees.
            if event.key == pygame.K_UP:
                self.value = min(self.max_val, self.value + step)
            else:
                self.value = max(self.min_val, self.value - step)


class Button:
    """Un bouton cliquable de la colonne de droite.

    Deux comportements selon le parametre switch :
      - bouton simple (switch=False)    : declenche une action une fois, puis revient
                                          au repos. Exemple : "Tout effacer".
      - interrupteur (switch=True)      : bascule entre actif et inactif a chaque
                                          clic et le reste. Exemple : "Dessiner"."""

    def __init__(self, text: str, command: Callable[[], None], switch: bool = False):
        """Prepare un bouton.

        Arguments :
          text    : le texte affiche sur le bouton. Il sera automatiquement coupe en
                    plusieurs lignes s'il est trop long pour la largeur disponible.
          command : la fonction a executer quand on agit sur le bouton. On passe ici
                    la fonction elle-meme, sans parentheses, pour que le bouton
                    puisse l'appeler plus tard.
          switch  : False pour un bouton d'action, True pour un interrupteur."""
        self.rect = pygame.Rect(0, 0, 0, 0)  # position a l'ecran, recalculee a chaque affichage
        self.text = text
        self.command = command
        self.switch = switch
        self.timer_active = 0                # instant, en millisecondes, ou l'action doit partir
        self.locked = False                  # True = bouton grise, les clics sont ignores
        self.active = False                  # interrupteur enfonce, ou bouton en cours de clic

    def draw(self, surface: pygame.Surface, index: int, total: int,
             column_ratio: float, font: pygame.font.Font) -> None:
        """Dessine le bouton, memorise sa position, et declenche l'action differee des boutons simples.

        Arguments :
          surface      : la fenetre sur laquelle dessiner.
          index        : le rang du bouton dans la colonne, a partir de 0, qui
                         determine sa hauteur d'affichage.
          total        : le nombre de boutons de l'ecran ; ils se partagent la
                         hauteur de la fenetre.
          column_ratio : largeur de la colonne en fraction de la fenetre.
          font         : la police du texte.

        Cette methode ne fait pas qu'afficher : c'est elle qui execute reellement la
        commande d'un bouton simple, une fois le delai d'enfoncement ecoule. Ce choix
        donne un retour visuel avant l'action, mais implique que rien ne se passe si
        le bouton n'est pas redessine."""
        if self.locked:
            color, thickness = (100, 100, 100), 1        # grise et discret
        elif self.active:
            color, thickness = (255, 255, 255), 4        # blanc et epais : enfonce
            # Pour un bouton simple, on a montre l'enfoncement pendant environ
            # 150 ms ; le delai est ecoule, on execute enfin la commande et on
            # relache le bouton. get_ticks donne le temps en millisecondes depuis
            # le demarrage de pygame.
            if not self.switch and pygame.time.get_ticks() > self.timer_active:
                self.command()
                self.active = False
        else:
            color, thickness = (200, 200, 200), 2        # etat normal

        # Les boutons sont empiles verticalement dans la colonne de droite et se
        # partagent equitablement la hauteur de la fenetre.
        win_w, win_h = pygame.display.get_surface().get_size()
        button_w = int(win_w * column_ratio)
        button_h = win_h / total
        self.rect = pygame.Rect(win_w - button_w, index * button_h, button_w, button_h)

        pygame.draw.rect(surface, (40, 40, 40), self.rect)        # fond
        pygame.draw.rect(surface, color, self.rect, thickness)    # cadre
        self._draw_multiline_text(surface, font, color)

    def _draw_multiline_text(self, surface, font, color) -> None:
        """Ecrit le texte du bouton en le coupant en lignes qui tiennent dans sa largeur.

        Arguments :
          surface : la fenetre sur laquelle ecrire.
          font    : la police, qui sert aussi a mesurer la largeur du texte.
          color   : la couleur du texte, choisie par draw selon l'etat du bouton.

        Ne sert qu'a draw. Sans elle, un texte comme "Retourner dessiner" deborderait
        du bouton au lieu de passer a la ligne."""
        # On ajoute les mots un par un tant que la ligne tient, puis on coupe.
        words = self.text.split(" ")
        lines: list[str] = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            # font.size renvoie le couple (largeur, hauteur) qu'occuperait ce texte
            # une fois affiche ; on ne garde que la largeur.
            if font.size(test_line)[0] < self.rect.width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)

        # On centre verticalement le bloc de lignes dans le bouton.
        # get_linesize donne la hauteur d'une ligne pour cette police.
        total_h = len(lines) * font.get_linesize()
        start_y = self.rect.centery - total_h // 2
        for i, line in enumerate(lines):
            text = font.render(line.strip(), True, color)
            text_rect = text.get_rect(centerx=self.rect.centerx,
                                      y=start_y + i * font.get_linesize())
            surface.blit(text, text_rect)

    def clicked(self, pos: tuple[int, int]) -> bool:
        """Traite un clic de souris et indique s'il concernait ce bouton.

        Arguments :
          pos : la position du clic en pixels, sous forme (x, y).

        Renvoie True si le clic est tombe sur le bouton, False sinon. L'ecran
        transmet chaque clic a tous ses boutons, et chacun decide lui-meme s'il est
        concerne."""
        if self.locked:
            return False
        if self.rect.collidepoint(pos):
            if self.switch:
                self.active = not self.active     # interrupteur : on bascule l'etat
                self.command()                    # et l'action part immediatement
            else:
                self.active = True                # bouton simple : on affiche l'enfoncement
                # et on programme l'action pour dans 150 ms ; c'est draw qui la lancera.
                self.timer_active = pygame.time.get_ticks() + 150
            return True
        return False


# =========================
# Base des ecrans
# =========================

class BaseScreen:
    """La classe mere des deux ecrans, qui rassemble tout ce qu'ils ont en commun.

    Elle contient le decoupage de la fenetre en zones, l'aiguillage des evenements,
    l'affichage de l'interface commune et la conversion de coordonnees. DrawScreen et
    AnimateScreen en HERITENT : elles disposent gratuitement de toutes ces methodes
    et ne reecrivent que ce qui leur est propre.

    Certaines methodes, comme update et progress, sont volontairement vides ou
    renvoient None. Elles servent de points d'extension : la classe mere les appelle,
    et chaque ecran decide de les redefinir ou non."""

    caption = ""   # titre de la fenetre ; chaque ecran concret donne le sien

    def __init__(self, app: "App"):
        """Prepare un ecran.

        Arguments :
          app : l'objet App, c'est-a-dire l'application principale. L'ecran en tire
                la fenetre sur laquelle dessiner, l'etat partage et la police. Les
                guillemets autour de "App" sont necessaires car cette classe est
                definie plus bas dans le fichier et n'existe pas encore ici."""
        self.app = app
        self.state = app.state         # raccourci vers l'etat partage
        self.surface = app.surface     # la fenetre
        self.font = app.font
        self.buttons: list[Button] = []            # rempli par chaque ecran concret
        self.value_boxes: list[ValueBox] = []      # idem ; l'ecran de dessin n'en a aucune
        self.next_screen: Optional[ScreenName] = None  # ecran demande, None pour rester ici
        self.layout = self.compute_layout()

    def compute_layout(self) -> Layout:
        """Decoupe la fenetre en trois zones et renvoie le resultat.

        La zone de dessin doit conserver un rapport largeur/hauteur fixe, pour que le
        dessin ne soit jamais deforme. Sa largeur etant imposee par la colonne de
        boutons, c'est la hauteur du bandeau du haut qui absorbe la difference.

        Appelee a chaque redimensionnement de la fenetre et a chaque deplacement de
        la bordure de la colonne."""
        win_w, win_h = self.surface.get_size()
        col_w = int(win_w * self.state.column_ratio)

        # Hauteur que devrait avoir le bandeau pour que la zone restante respecte
        # exactement le rapport voulu. Elle peut sortir des limites de la fenetre, on
        # la ramene donc ensuite entre 10 pixels et la hauteur moins 10.
        theoretical_header_h = win_h - (win_w - col_w) / DRAWING_ZONE_RATIO
        header_at_min = theoretical_header_h <= 10
        header_h = max(10, min(win_h - 10, theoretical_header_h))

        rect_col = pygame.Rect(win_w - col_w, 0, col_w, win_h)                 # colonne de droite
        rect_info = pygame.Rect(0, 0, win_w - col_w, header_h)                 # bandeau du haut
        rect_main = pygame.Rect(0, header_h, win_w - col_w, win_h - header_h)  # zone centrale
        return Layout(rect_col, rect_info, rect_main, header_at_min)

    def on_enter(self) -> None:
        """Remet l'ecran dans son etat de depart. Appelee a chaque arrivee sur cet ecran.

        Efface la demande de changement d'ecran, recalcule la mise en page au cas ou
        la fenetre aurait change de taille pendant qu'on etait ailleurs, et met a jour
        le titre de la fenetre. AnimateScreen redefinit cette methode pour y ajouter le
        calcul des coefficients de Fourier."""
        self.next_screen = None
        self.layout = self.compute_layout()
        pygame.display.set_caption(self.caption)

    def to_screen(self, p: tuple[float, float]) -> tuple[float, float]:
        """Convertit un point du dessin en position a l'ecran.

        Arguments :
          p : un point en coordonnees logiques, c'est-a-dire avec l'abscisse dans
              [0 ; 1.5] et l'ordonnee dans [0 ; 1], quelle que soit la taille de la
              fenetre.

        Renvoie la position correspondante en pixels, dans la zone de dessin.

        Tout le programme stocke et calcule en coordonnees logiques, et ne convertit
        qu'au dernier moment, pour l'affichage. Cela permet au meme dessin de
        s'afficher correctement a n'importe quelle taille de fenetre, et garde les
        calculs mathematiques independants de la resolution de l'ecran."""
        return (
            self.layout.rect_main.x + (p[0] / DRAWING_ZONE_RATIO) * self.layout.rect_main.width,
            self.layout.rect_main.y + p[1] * self.layout.rect_main.height,
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        """Aiguille un evenement vers le traitement qui lui correspond.

        Arguments :
          event : un evenement pygame : clic, touche, redimensionnement, fermeture.

        Chaque evenement est d'abord transmis aux boites de reglage, puis trie sur son
        type. C'est le point d'entree unique de toutes les actions de l'utilisateur."""
        for value_box in self.value_boxes:
            value_box.handle_event(event)

        if event.type == pygame.QUIT:              # clic sur la croix de fermeture
            self.next_screen = ScreenName.QUIT
        elif event.type == pygame.VIDEORESIZE:     # la fenetre vient d'etre redimensionnee
            # set_mode doit etre rappele pour obtenir une surface a la nouvelle taille ;
            # l'ancienne n'est plus valide. On la redistribue a l'application et a l'ecran.
            self.app.surface = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            self.surface = self.app.surface
            self.layout = self.compute_layout()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.state.border_drag = False         # on lache la bordure de la colonne
        elif event.type == pygame.MOUSEMOTION and self.state.border_drag:
            self._handle_border_drag()

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        """Decide ce que declenche un clic : saisir la bordure, ou actionner un bouton.

        Arguments :
          event : l'evenement de clic, dont on utilise la position.

        Ne sert qu'a handle_event. La bordure est prioritaire : si le clic tombe a
        moins de huit pixels d'elle, on entre en mode deplacement et on ignore les
        boutons, sans quoi il serait impossible de saisir la bordure quand un bouton
        se trouve juste derriere."""
        mouse_x, _ = pygame.mouse.get_pos()
        win_w, _ = self.surface.get_size()
        border_x = win_w - win_w * self.state.column_ratio
        if abs(mouse_x - border_x) < 8:
            self.state.border_drag = True
            return
        # Sinon on transmet le clic a chaque bouton ; chacun teste s'il est concerne.
        for button in self.buttons:
            button.clicked(event.pos)

    def _handle_border_drag(self) -> None:
        """Elargit ou retrecit la colonne de boutons en suivant la souris.

        Ne sert qu'a handle_event, et uniquement pendant que border_drag vaut True.
        La largeur est bornee entre un dixieme et la moitie de la fenetre. La
        condition finale empeche d'elargir davantage la colonne quand le bandeau du
        haut est deja ecrase au minimum, mais autorise toujours a la retrecir."""
        mouse_x, _ = pygame.mouse.get_pos()
        win_w, _ = self.surface.get_size()
        new_ratio = max(0.1, min(0.5, (win_w - mouse_x) / win_w))
        if not self.layout.header_at_min or new_ratio > self.state.column_ratio:
            self.state.column_ratio = new_ratio
            self.layout = self.compute_layout()

    def update(self) -> None:
        """Met a jour la logique interne de l'ecran, une fois par image.

        Vide dans la classe mere : c'est un point d'extension que chaque ecran
        redefinit. L'ecran de dessin y lit la souris pour tracer, l'ecran d'animation
        y fait avancer le temps."""
        pass

    def progress(self) -> Optional[float]:
        """Indique l'avancement a afficher dans la barre de progression, entre 0 et 1.

        Renvoie None dans la classe mere, ce qui signifie que cet ecran n'affiche
        aucune barre. AnimateScreen redefinit cette methode pour renvoyer le parametre
        de temps de l'animation. C'est le meme mecanisme que update : la classe mere
        interroge, chaque ecran repond ce qu'il veut."""
        return None

    def _draw_progress_bar(self, height: int) -> None:
        """Dessine la barre de progression au bas du bandeau du haut.

        Arguments :
          height : la hauteur de la barre en pixels, decidee par draw_base_ui.

        Ne sert qu'a draw_base_ui, et seulement quand progress renvoie une valeur.
        La barre n'est pas un objet a part entiere : n'ayant ni etat a conserver ni
        clic a traiter, elle se reduit a trois rectangles redessines a chaque image."""
        # On borne la valeur par securite, au cas ou progress renverrait un nombre
        # legerement hors des limites a cause des arrondis.
        value = max(0.0, min(1.0, self.progress()))
        rect = pygame.Rect(self.layout.rect_info.x,
                           self.layout.rect_info.bottom - height,   # .bottom = bas du bandeau
                           self.layout.rect_info.width, height)
        pygame.draw.rect(self.surface, (30, 30, 30), rect)          # le fond, sur toute la largeur
        pygame.draw.rect(self.surface, (255, 255, 0),               # la part parcourue, en jaune
                         pygame.Rect(rect.x, rect.y, int(rect.width * value), rect.height))
        pygame.draw.rect(self.surface, (100, 100, 100), rect, 1)    # le cadre

    def draw_base_ui(self) -> None:
        """Efface l'ecran et dessine tout ce qui est commun aux deux ecrans.

        C'est-a-dire le fond noir, les boutons de la colonne, la boite de reglage
        eventuelle, la barre de progression eventuelle, et les deux traits gris qui
        separent les trois zones. Chaque ecran appelle cette methode en premier, puis
        ajoute par-dessus ce qui lui est propre."""
        self.surface.fill((0, 0, 0))

        if self.layout.rect_col.width > 20:      # colonne trop etroite : on n'affiche rien
            for index, button in enumerate(self.buttons):
                button.draw(self.surface, index, len(self.buttons),
                            self.state.column_ratio, self.font)

        # La barre de progression occupe le bas du bandeau, la boite de reglage se
        # partage ce qui reste au-dessus. On n'affiche la barre que si le bandeau est
        # assez haut pour loger les deux sans que l'une ecrase l'autre.
        bar_h = PROGRESS_BAR_H if (self.progress() is not None
                                   and self.layout.rect_info.height > 30) else 0
        boxes_rect = pygame.Rect(self.layout.rect_info.x, self.layout.rect_info.y,
                                 self.layout.rect_info.width,
                                 self.layout.rect_info.height - bar_h)

        # En passant ce rectangle raccourci aux boites, on raccourcit du meme coup
        # leur zone cliquable, puisqu'elles en deduisent leur position.
        if self.value_boxes and boxes_rect.height > 15:
            for index, value_box in enumerate(self.value_boxes):
                value_box.draw(self.surface, index, len(self.value_boxes),
                               boxes_rect, self.font)

        if bar_h:
            self._draw_progress_bar(bar_h)

        # Les deux traits de separation entre les trois zones.
        pygame.draw.line(self.surface, (100, 100, 100),
                         (self.layout.rect_col.x, 0),
                         (self.layout.rect_col.x, self.surface.get_height()), 2)
        pygame.draw.line(self.surface, (100, 100, 100),
                         (0, self.layout.rect_info.height),
                         (self.layout.rect_info.width, self.layout.rect_info.height), 2)

    def draw(self) -> None:
        """Dessine l'ecran complet et l'affiche.

        Version minimale de la classe mere : seule l'interface commune est dessinee.
        Les deux ecrans concrets redefinissent cette methode pour ajouter le dessin ou
        l'animation avant l'affichage."""
        self.draw_base_ui()
        # Tout ce qui precede a ete dessine en memoire, invisible. flip transfere d'un
        # bloc le contenu de la surface vers l'ecran physique.
        pygame.display.flip()


# =========================
# Outils Fourier
# =========================

class FourierEngine:
    """Les calculs lies a Fourier, sans aucun lien avec l'affichage.

    Cette classe prend le trace produit par GeometryEngine et le transforme en une
    liste de cercles tournants. Elle procede en deux temps : d'abord preparer les
    points pour qu'ils remplissent les conditions de la theorie, ensuite calculer la
    transformee de Fourier discrete.

    Toutes ses methodes sont des @staticmethod : ce sont des fonctions ordinaires,
    simplement rangees dans la classe pour les regrouper. On les appelle par
    FourierEngine.compute_dft(...), sans jamais creer d'objet FourierEngine, et elles
    ne recoivent pas de parametre self. Aucune n'utilise pygame, ce qui permet de les
    tester independamment de l'interface."""

    CLOSE_TOL = 1e-6   # en dessous de cette distance, on considere que le trace est ferme

    @staticmethod
    def close_by_backtrack(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Rend le trace ferme, en lui ajoutant au besoin son propre parcours inverse.

        Arguments :
          points : le trace a examiner, sous forme de liste de points dans l'ordre
                   du parcours.

        Renvoie le trace inchange s'il revient deja a son point de depart, ou bien le
        trace suivi de son parcours en sens inverse.

        POURQUOI C'EST NECESSAIRE. La transformee de Fourier discrete considere les
        points qu'on lui donne comme UNE periode d'un signal qui se repete
        indefiniment : apres le dernier point, elle enchaine sur le premier. Si le
        trace est ouvert, elle suppose donc un saut brutal entre les deux bouts. Or
        une somme de sinusoides ne sait pas sauter : elle produit a la place un
        segment droit traversant le dessin, accompagne d'oscillations qui se
        propagent tout le long de la courbe.

        QUAND ELLE AGIT REELLEMENT. Dans le cas courant, GeometryEngine a deja produit
        un circuit ferme et cette methode ne fait rien. Elle sert de filet de securite
        pour le seul cas qui lui echappe : un dessin constitue d'un unique trait, que
        le programme utilise directement sans le faire passer par le graphe.

        UNE CONSEQUENCE INTERESSANTE. Le trace obtenu par aller-retour est un
        palindrome : le point numero k et le point numero N-1-k sont le meme. On en
        deduit que les coefficients de frequences opposees sont egaux, et comme
        c*exp(i*a) + c*exp(-i*a) vaut 2*c*cos(a), chaque paire de cercles cesse de
        tourner pour osciller le long d'une direction fixe. Le mouvement d'ensemble
        devient une somme d'oscillations, qui repasse forcement par ou elle est
        venue : l'epicycloide fait alors un vrai aller-retour sur le trace."""
        if math.hypot(points[-1][0] - points[0][0],
                      points[-1][1] - points[0][1]) <= FourierEngine.CLOSE_TOL:
            return points
        # points[-2::-1] se lit ainsi : partir de l'avant-dernier point, aller vers
        # le debut, un point a la fois. On saute volontairement le dernier point, qui
        # figure deja a la fin de la liste, pour ne pas le repeter au demi-tour.
        return points + points[-2::-1]

    @staticmethod
    def uniformize(paths: list[list[tuple[float, float]]], nb_points: int) -> Optional[np.ndarray]:
        """Remplace les points du trace par un nombre fixe de points regulierement espaces.

        Arguments :
          paths     : le trace, enveloppe dans une liste d'un seul element pour
                      rester compatible avec le format employe ailleurs dans le
                      programme.
          nb_points : le nombre de points voulus en sortie, soit NB_SAMPLES.

        Renvoie un tableau numpy de forme (nb_points, 2), ou None si le trace fourni
        est inutilisable.

        POURQUOI C'EST NECESSAIRE. La formule de la transformee suppose des points
        regulierement espaces dans le temps. Or les points sont enregistres au rythme
        de la souris : ils sont serres la ou l'utilisateur a ralenti et clairsemes la
        ou il est alle vite. Utilises tels quels, les points serres compteraient trop
        et fausseraient les coefficients.

        LA METHODE. On repere chaque point non par son numero mais par la distance
        parcourue depuis le debut du trace, appelee abscisse curviligne. On decide
        ensuite des distances regulierement espacees, et on cherche quel point du
        trace se trouve a chacune de ces distances."""
        if len(paths) != 1 or len(paths[0]) < 2:    # il faut un seul trace, d'au moins deux points
            return None

        # Fermeture eventuelle du trace, avant toute autre chose.
        raw_points = FourierEngine.close_by_backtrack(list(paths[0]))

        points = np.array(raw_points, dtype=float)  # tableau de forme (m, 2) : m lignes, x et y
        # np.diff soustrait chaque ligne a la suivante : on obtient les vecteurs qui
        # relient chaque point au suivant. axis=0 precise qu'on travaille ligne par ligne.
        diffs = np.diff(points, axis=0)
        # np.hypot applique Pythagore a deux tableaux : on obtient la longueur de
        # chaque petit segment du trace.
        distances = np.hypot(diffs[:, 0], diffs[:, 1])

        # Abscisse curviligne : la case k contient la longueur parcourue du debut
        # jusqu'au point k. np.cumsum accumule les valeurs de proche en proche.
        # La premiere case reste a zero, d'ou l'ecriture [1:] qui remplit a partir
        # de la deuxieme.
        cumulative_distances = np.zeros(len(points))
        cumulative_distances[1:] = np.cumsum(distances)

        total_distance = cumulative_distances[-1]   # longueur totale du trace
        if total_distance == 0:                     # tous les points confondus : rien a faire
            return None

        # np.linspace fabrique nb_points valeurs regulierement espacees entre 0 et la
        # longueur totale. Ce sont les distances auxquelles on veut un point.
        target_distances = np.linspace(0, total_distance, nb_points)
        # np.interp repond a la question : "a telle distance du debut, ou se trouve le
        # crayon ?". Elle localise entre quels deux points connus tombe la distance
        # demandee, et interpole lineairement entre eux. On l'appelle separement pour
        # l'abscisse et pour l'ordonnee, avec les memes distances, donc les deux
        # resultats se correspondent point par point.
        new_x = np.interp(target_distances, cumulative_distances, points[:, 0])
        new_y = np.interp(target_distances, cumulative_distances, points[:, 1])
        # np.column_stack recolle les deux listes en un tableau a deux colonnes.
        return np.column_stack((new_x, new_y))

    @staticmethod
    def compute_dft(points: np.ndarray) -> list[dict[str, float]]:
        """Calcule les cercles tournants dont la somme redessine le trace.

        Arguments :
          points : le tableau de points regulierement espaces produit par uniformize,
                   de forme (N, 2).

        Renvoie la liste des coefficients, chacun sous forme d'un dictionnaire a trois
        cles : freq pour la vitesse de rotation, amp pour le rayon du cercle, phase
        pour son angle de depart. La liste est triee par rayon decroissant.

        L'IDEE. On identifie chaque point du plan au nombre complexe x + i*y. Un
        cercle parcouru a vitesse constante s'ecrit alors A*exp(i*(2*pi*n*t + phi)) :
        A est le rayon, phi l'angle de depart, n le nombre de tours effectues pendant
        que t va de 0 a 1. La question de Fourier est de trouver quels rayons et
        quelles phases choisir pour que la somme de ces cercles dessine la courbe.

        POURQUOI LA FORMULE MARCHE. Pour isoler le cercle de frequence n, on multiplie
        tout le signal par exp(-2i*pi*n*t) : cela fait tourner le plan en sens inverse
        a la vitesse n, si bien que le cercle recherche devient immobile tandis que
        tous les autres continuent de tourner. On prend ensuite la moyenne : celle
        d'un point immobile est ce point lui-meme, tandis que celle d'un point qui
        tourne a vitesse constante sur un tour complet est nulle, par symetrie. Il ne
        reste donc que le cercle vise."""
        n_points = len(points)
        # On assemble les deux colonnes en un seul tableau de nombres complexes. En
        # Python, 1j designe l'unite imaginaire.
        z = points[:, 0] + 1j * points[:, 1]
        # np.arange(N) donne 0, 1, ..., N-1 ; divise par N, on obtient le parametre de
        # temps de chaque point, reparti regulierement dans [0 ; 1[.
        t_steps = np.arange(n_points) / n_points

        coeffs = []
        # Les frequences vont de -N/2 a N/2-1. Les negatives correspondent aux cercles
        # qui tournent dans l'autre sens, indispensables ici car le signal est complexe
        # et non reel.
        for freq in range(-n_points // 2, n_points // 2):
            # np.mean calcule la moyenne d'un tableau, ce qui realise exactement le
            # facteur (1/N) multiplie par la somme de la formule.
            c_n = np.mean(z * np.exp(-1j * 2 * np.pi * freq * t_steps))
            coeffs.append({
                "freq": freq,
                "amp": float(np.abs(c_n)),          # np.abs d'un complexe donne son module
                "phase": float(np.angle(c_n)),      # np.angle donne son argument, en radians
            })
        # Tri par rayon decroissant. C'est purement visuel : les grands cercles
        # dessines en premier donnent une animation plus lisible. key indique sur quoi
        # comparer, reverse=True inverse l'ordre.
        return sorted(coeffs, key=lambda c: c["amp"], reverse=True)


# =========================
# Outils geometriques
# =========================

class GeometryEngine:
    """Transforme un dessin fait de plusieurs traits en une seule courbe fermee.

    C'est la partie la plus complexe du programme. Elle procede en deux temps :
      1) relier les traits separes par des segments artificiels, appeles ponts,
         jusqu'a ce que le dessin ne forme plus qu'un seul morceau ;
      2) modeliser le tout comme un graphe, et y chercher un parcours qui emprunte
         chaque segment et revient a son point de depart, en repassant le moins
         possible sur ses pas. C'est le probleme dit du postier chinois.

    Comme FourierEngine, elle n'est faite que de @staticmethod et n'utilise pas
    pygame."""

    KEY_PRECISION = 9   # nombre de decimales retenues pour decider que deux points sont le meme

    @staticmethod    # une fonction rangee dans la classe, sans parametre self
    def _key(point: tuple[float, float]) -> tuple[float, float]:
        """Fabrique la carte d'identite d'un point, en arrondissant ses coordonnees.

        Arguments :
          point : le point a identifier.

        Ne sert qu'aux methodes de cette classe. Deux points sont consideres comme le
        meme sommet du graphe si, et seulement si, cette cle est identique. L'arrondi
        a neuf decimales absorbe les minuscules erreurs des nombres a virgule
        flottante, qui feraient sinon passer pour distincts deux points censes
        coincider.

        Consequence importante : le programme ne detecte pas les intersections. Deux
        traits qui se croisent a l'ecran ne partagent presque jamais un point aux
        coordonnees exactement identiques, et restent donc separes pour le graphe."""
        return (round(point[0], GeometryEngine.KEY_PRECISION),
                round(point[1], GeometryEngine.KEY_PRECISION))

    @staticmethod
    def _poly_length(points: list[tuple[float, float]]) -> float:
        """Calcule la longueur totale d'une suite de points relies par des segments.

        Arguments :
          points : la suite de points, dans l'ordre du parcours.

        Sert a donner un poids aux aretes du graphe : la longueur d'une arete est la
        distance que le crayon parcourt reellement en la suivant. C'est ce poids que
        Dijkstra additionne pour comparer les chemins."""
        return sum(math.hypot(points[i + 1][0] - points[i][0],
                              points[i + 1][1] - points[i][1])
                   for i in range(len(points) - 1))

    @staticmethod
    def _avg_spacing(strokes: list[list[tuple[float, float]]]) -> float:
        """Calcule la distance moyenne entre deux points consecutifs du dessin.

        Arguments :
          strokes : tous les traits du dessin.

        Ne sert qu'a connect_paths, qui transmet ce nombre a _sample_segment : les
        ponts ajoutes recoivent ainsi la meme densite de points que le reste du
        dessin. La valeur de repli 0.01 evite une division par zero sur un dessin
        vide."""
        total, count = 0.0, 0
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                total += math.hypot(stroke[i + 1][0] - stroke[i][0],
                                    stroke[i + 1][1] - stroke[i][1])
                count += 1
        return total / count if count else 0.01

    @staticmethod
    def _sample_segment(p_from, p_to, spacing: float) -> list[tuple[float, float]]:
        """Decoupe un segment droit en points regulierement espaces, extremites comprises.

        Arguments :
          p_from  : le point de depart du segment.
          p_to    : son point d'arrivee.
          spacing : l'espacement souhaite entre deux points consecutifs, fourni par
                    _avg_spacing.

        Ne sert qu'a connect_paths, pour fabriquer les ponts. Le resultat est
        equivalent, pour l'algorithme de parcours comme pour Fourier, a un pont reduit
        a ses deux extremites : les points intermediaires sont de toute facon
        supprimes par la contraction du graphe. Leur utilite est ailleurs, dans
        l'interface : ils rendent le pont visible avec l'option d'affichage des
        points, et permettent a la gomme d'agir en son milieu.

        Un detail a son importance : le dernier point calcule retombe exactement sur
        p_to, ce qui permet au pont de se souder au trait qu'il vise, puisque
        l'identite des sommets se fait par coordonnees."""
        length = math.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1])
        # Nombre de morceaux a decouper. max(1, ...) garantit au moins un morceau,
        # donc deux points, meme si le pont est plus court que l'espacement voulu.
        steps = max(1, int(round(length / spacing))) if spacing > 0 else 1
        # Pour i allant de 0 a steps, on avance de la fraction i/steps le long du
        # segment. A i valant 0 on est sur p_from, a i valant steps sur p_to : d'ou
        # steps + 1 points au total.
        return [(p_from[0] + (i / steps) * (p_to[0] - p_from[0]),
                 p_from[1] + (i / steps) * (p_to[1] - p_from[1]))
                for i in range(steps + 1)]

    @staticmethod
    def _subsample(points: np.ndarray, cap: int) -> np.ndarray:
        """Ne garde qu'un nombre limite de points, repartis regulierement dans le tableau.

        Arguments :
          points : le tableau de points a reduire.
          cap    : le nombre maximal de points a conserver.

        Ne sert qu'a _nearest_pair, pour eviter une explosion de la memoire. Cette
        derniere compare toutes les paires de points entre deux morceaux du dessin :
        avec deux morceaux de six mille points, le calcul mobiliserait plus de cinq
        cents megaoctets d'un coup. En plafonnant a huit cents points de chaque cote,
        on retombe a une dizaine de megaoctets.

        Le prix a payer est negligeable : on ne trouve plus forcement le couple
        exactement le plus proche, mais seulement le plus proche parmi les points
        retenus, ce qui decale le pont de quelques millioniemes."""
        if len(points) <= cap:      # deja assez petit : on renvoie tel quel
            return points
        # np.linspace donne cap positions regulierement reparties entre 0 et le
        # dernier indice ; .astype(int) les arrondit en entiers utilisables comme
        # indices. points[idx] selectionne d'un coup toutes les lignes voulues.
        idx = np.linspace(0, len(points) - 1, cap).astype(int)
        return points[idx]

    @staticmethod
    def _nearest_pair(set_a: np.ndarray, set_b: np.ndarray):
        """Trouve le couple de points les plus proches entre deux ensembles.

        Arguments :
          set_a : les points d'un premier morceau du dessin, tableau de forme (n, 2).
          set_b : les points d'un second morceau, tableau de forme (m, 2).

        Renvoie trois choses : le point retenu dans set_a, celui retenu dans set_b, et
        la distance qui les separe.

        Ne sert qu'a connect_paths, qui s'en sert pour decider ou poser un pont et
        quelles composantes relier en priorite. La distance calculee ici est a vol
        d'oiseau, et non le long des traits, ce qui est correct puisqu'un pont est
        precisement un segment droit traversant le vide."""
        # On plafonne d'abord la taille des deux ensembles.
        a = GeometryEngine._subsample(set_a, 800)
        b = GeometryEngine._subsample(set_b, 800)
        # Cette ligne calcule d'un coup l'ecart entre CHAQUE point de a et CHAQUE
        # point de b. Le mot-cle None insere une dimension vide : a passe de la forme
        # (n, 2) a (n, 1, 2), et b de (m, 2) a (1, m, 2). Numpy etire alors
        # automatiquement les dimensions de taille 1 pour les faire correspondre,
        # mecanisme appele diffusion, et soustrait terme a terme.
        # Le resultat diff a la forme (n, m, 2) : diff[i][j] contient l'ecart entre le
        # i-ieme point de a et le j-ieme point de b, sous forme (ecart en x, ecart en y).
        diff = a[:, None, :] - b[None, :, :]
        # On en deduit toutes les distances au carre. On evite la racine carree, car
        # comparer des carres revient au meme et coute moins cher.
        dist2 = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
        # np.argmin renvoie la position du minimum, mais en numerotant les cases comme
        # si le tableau etait une seule longue ligne.
        flat = int(np.argmin(dist2))
        # divmod redonne les deux indices : le quotient est la ligne, le reste la
        # colonne. dist2.shape[1] est le nombre de colonnes, c'est-a-dire la taille de b.
        i, j = divmod(flat, dist2.shape[1])
        pa = (float(a[i, 0]), float(a[i, 1]))
        pb = (float(b[j, 0]), float(b[j, 1]))
        return pa, pb, math.sqrt(float(dist2[i, j]))

    @staticmethod
    def connect_paths(paths: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        """Relie les traits separes du dessin par des ponts, jusqu'a n'avoir plus qu'un morceau.

        Arguments :
          paths : les traits du dessin, tels que l'utilisateur les a traces.

        Renvoie la liste des traits d'origine, suivie de la liste des ponts ajoutes.
        Chaque pont est lui-meme une suite de points, donc un trait comme les autres.

        POURQUOI. Si le dessin est en plusieurs morceaux, aucun parcours ne peut tous
        les visiter : il faudrait sauter dans le vide. On ajoute donc des segments
        artificiels, qui feront desormais partie du dessin et seront visibles a
        l'animation.

        LA METHODE. A chaque tour, on cherche les deux morceaux les plus proches et on
        les relie par le segment le plus court possible entre eux. En choisissant
        toujours le pont le plus court disponible, on obtient un arbre couvrant
        minimal : la longueur totale de trait artificiel ajoutee est la plus petite
        possible."""
        strokes = [list(p) for p in paths if len(p) >= 1]
        if len(strokes) <= 1:            # un seul morceau : il n'y a rien a relier
            return [list(p) for p in paths]

        spacing = GeometryEngine._avg_spacing(strokes)

        # --- Union-find : une structure qui repond a la question "ces deux traits
        # sont-ils deja dans le meme morceau ?". Chaque trait pointe vers un chef de
        # groupe ; deux traits sont dans le meme groupe s'ils ont le meme chef. ---
        parent = list(range(len(strokes)))   # au depart, chaque trait est son propre chef

        def find(a: int) -> int:
            """Remonte jusqu'au chef du groupe de a et le renvoie."""
            while parent[a] != a:
                # Compression de chemin : au passage, on raccourcit la chaine en
                # faisant pointer a directement vers son grand-parent. Les remontees
                # suivantes deviennent quasi instantanees.
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            """Fusionne les groupes de a et de b, en faisant pointer un chef vers l'autre."""
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Deux traits qui partagent deja un point exactement identique sont dans le
        # meme morceau des le depart : on les fusionne avant toute chose. Le
        # dictionnaire seen retient, pour chaque point rencontre, le trait ou on l'a
        # vu la premiere fois.
        seen: dict[tuple[float, float], int] = {}
        for i, stroke in enumerate(strokes):
            for point in stroke:
                k = GeometryEngine._key(point)
                if k in seen:
                    union(i, seen[k])
                else:
                    seen[k] = i

        bridges: list[list[tuple[float, float]]] = []
        while True:
            # On regroupe les traits par morceau : la cle est le chef, la valeur la
            # liste des traits qui lui appartiennent. setdefault cree la liste vide
            # au premier passage.
            comps: dict[int, list[int]] = {}
            for i in range(len(strokes)):
                comps.setdefault(find(i), []).append(i)
            if len(comps) <= 1:              # tout est relie : c'est termine
                break

            # On rassemble les points de chaque morceau dans un seul tableau, pour
            # pouvoir les comparer d'un bloc.
            comp_points: dict[int, np.ndarray] = {}
            for root, indices in comps.items():
                pts: list[tuple[float, float]] = []
                for i in indices:
                    pts.extend(strokes[i])
                comp_points[root] = np.array(pts, dtype=float)

            # On essaie toutes les paires de morceaux et on retient celle dont les
            # points les plus proches sont les moins eloignes.
            roots = list(comps.keys())
            best = None
            for a in range(len(roots)):
                for b in range(a + 1, len(roots)):
                    pa, pb, dist = GeometryEngine._nearest_pair(comp_points[roots[a]],
                                                                comp_points[roots[b]])
                    if best is None or dist < best[0]:
                        best = (dist, roots[a], roots[b], pa, pb)

            # On ajoute le pont correspondant et on declare les deux morceaux reunis.
            _, ra, rb, pa, pb = best
            bridges.append(GeometryEngine._sample_segment(pa, pb, spacing))
            union(ra, rb)

        return strokes + bridges

    @staticmethod
    def build_single_path(paths: list[list[tuple[float, float]]]) -> Optional[list[tuple[float, float]]]:
        """Transforme tout le dessin en une seule liste de points formant une boucle fermee.

        Arguments :
          paths : les traits du dessin, tels que l'utilisateur les a traces.

        Renvoie la suite ordonnee des points a parcourir, ou None si le dessin est
        inutilisable. Cette suite revient a son point de depart.

        C'est la methode principale de la classe, et le seul point d'entree utilise
        par le reste du programme. Elle enchaine cinq etapes, signalees dans le code :
          1) construire un graphe : un sommet par point, une arete entre deux points
             consecutifs d'un meme trait ;
          2) le contracter, en remplacant les longues suites de points sans
             embranchement par une seule arete, ce qui le reduit enormement ;
          3) equilibrer les degres, en dupliquant certaines aretes, pour qu'un
             parcours ferme devienne possible ;
          4) extraire ce parcours ;
          5) le redeployer en liste de points.

        POURQUOI L'ETAPE 3. Un parcours ferme qui emprunte chaque arete une seule fois
        n'existe que si tous les sommets ont un nombre PAIR d'aretes. La raison est
        simple : chaque fois que le parcours traverse un sommet, il y consomme deux
        aretes, une pour entrer et une pour sortir. Or un dessin a beaucoup de sommets
        impairs, ne serait-ce que les bouts de traits, qui n'ont qu'une seule arete.
        Dupliquer une arete revient a autoriser le crayon a repasser dessus, et
        dupliquer tout un chemin change la parite de ses deux extremites uniquement,
        les sommets intermediaires gagnant deux aretes chacun."""
        strokes = GeometryEngine.connect_paths([p for p in paths if len(p) >= 2])
        if not strokes:
            return None

        # --- 1) Construction du graphe. ---
        ids: dict[tuple[float, float], int] = {}        # coordonnees arrondies -> numero de sommet
        coords: list[tuple[float, float]] = []          # numero de sommet -> coordonnees
        edges: list[tuple[int, int]] = []               # liste des aretes, chacune = (sommet, sommet)
        inc: dict[int, list[int]] = defaultdict(list)   # sommet -> numeros des aretes qui le touchent

        def node_id(point: tuple[float, float]) -> int:
            """Renvoie le numero du sommet situe a ce point, en le creant s'il est nouveau.

            C'est ici que les traits se soudent entre eux : un point deja rencontre
            ailleurs recupere le meme numero, donc devient le meme sommet. C'est ainsi
            qu'un pont dont l'extremite tombe au milieu d'un trait s'y raccroche."""
            k = GeometryEngine._key(point)
            if k not in ids:
                ids[k] = len(coords)
                coords.append((float(point[0]), float(point[1])))
            return ids[k]

        for stroke in strokes:
            prev: Optional[int] = None      # remis a None a chaque trait, pour ne pas relier
            for point in stroke:            # la fin d'un trait au debut du suivant
                cur = node_id(point)
                if prev is not None and prev != cur:
                    ei = len(edges)
                    edges.append((prev, cur))
                    inc[prev].append(ei)
                    inc[cur].append(ei)
                prev = cur

        if not edges:
            return None

        # --- 2) Contraction du graphe. ---
        # Un dessin de cinq mille points donne cinq mille sommets, dont la quasi-totalite
        # sont de simples points de passage : une arete arrive, une arete repart. On
        # remplace chaque suite de tels sommets par une seule "super-arete", qui memorise
        # la liste complete des points traverses afin de pouvoir tout redeployer plus tard.
        # Il ne reste alors que les bouts de traits et les embranchements, soit une dizaine
        # de sommets au lieu de milliers.
        used = [False] * len(edges)
        super_edges: list[dict] = []   # chaque element : {a, b, pts (de a vers b), length}

        def walk_chain(start_edge: int, start_node: int) -> None:
            """Suit une suite de points sans embranchement et en fait une seule super-arete.

            Arguments :
              start_edge : le numero de l'arete par laquelle on commence.
              start_node : le sommet d'ou l'on part.

            Avance tant que les sommets rencontres n'ont exactement que deux aretes,
            c'est-a-dire tant qu'il n'y a pas de choix a faire, et s'arrete des qu'elle
            atteint un bout de trait ou un embranchement."""
            used[start_edge] = True
            a, b = edges[start_edge]
            nxt = b if a == start_node else a     # l'autre extremite de l'arete de depart
            pts = [coords[start_node], coords[nxt]]
            cur_edge, cur = start_edge, nxt
            while len(inc[cur]) == 2:             # exactement deux aretes : simple point de passage
                e0, e1 = inc[cur]
                nxt_edge = e1 if e0 == cur_edge else e0   # on prend l'autre arete que celle d'ou l'on vient
                if used[nxt_edge]:                # deja parcourue : on boucle, on s'arrete
                    break
                used[nxt_edge] = True
                x, y = edges[nxt_edge]
                nn = y if x == cur else x
                pts.append(coords[nn])
                cur_edge, cur = nxt_edge, nn
            super_edges.append({"a": start_node, "b": cur,
                                "pts": pts, "length": GeometryEngine._poly_length(pts)})

        # On lance walk_chain depuis chaque sommet interessant, c'est-a-dire chaque
        # sommet qui n'a pas exactement deux aretes, et le long de chacune de ses aretes.
        junctions = [n for n in range(len(coords)) if len(inc[n]) != 2]
        for junction in junctions:
            for ei in list(inc[junction]):
                if not used[ei]:
                    walk_chain(ei, junction)

        # Filet de securite pour les boucles fermees sur elles-memes, entierement
        # composees de points de passage : elles n'ont aucun sommet interessant, donc
        # la boucle precedente ne les a jamais atteintes.
        for ei in range(len(edges)):
            if not used[ei]:
                walk_chain(ei, edges[ei][0])

        if not super_edges:
            return None

        # --- 3) Postier chinois : rendre tous les degres pairs. ---
        # degree compte les aretes de chaque sommet dans le graphe contracte.
        # sadj est la liste d'adjacence : pour chaque sommet, la liste de ses voisins
        # sous forme (voisin, longueur du trajet, numero de la super-arete). C'est la
        # forme dont Dijkstra a besoin pour repondre vite a "ou puis-je aller d'ici,
        # et a quel prix".
        degree: dict[int, int] = defaultdict(int)
        sadj: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
        for idx, edge in enumerate(super_edges):
            degree[edge["a"]] += 1
            degree[edge["b"]] += 1
            # Chaque arete est inscrite dans les deux sens, puisqu'elle se parcourt
            # dans les deux sens.
            sadj[edge["a"]].append((edge["b"], edge["length"], idx))
            sadj[edge["b"]].append((edge["a"], edge["length"], idx))

        odd = [n for n, d in degree.items() if d % 2 == 1]   # les sommets a corriger

        def dijkstra(src: int):
            """Calcule le plus court chemin depuis un sommet vers tous les autres.

            Arguments :
              src : le sommet de depart.

            Renvoie trois dictionnaires : la distance de src a chaque sommet, l'arete
            empruntee pour atteindre chaque sommet, et le sommet precedent sur le
            chemin. Les deux derniers permettent de reconstituer le chemin en
            remontant de proche en proche.

            LE PRINCIPE. On tient une distance provisoire pour chaque sommet, et on
            fige toujours le plus petit non encore fige. Une fois fige, un sommet ne
            sera plus jamais remis en cause : tout autre chemin vers lui devrait
            passer par un sommet encore provisoire, donc deja plus loin, et comme
            toutes les longueurs sont positives, poursuivre ne peut qu'ajouter.

            Attention au vocabulaire : la distance dont il est question est toujours
            la longueur depuis src, mesuree le long des traits du dessin, et jamais
            la distance a vol d'oiseau ni la distance au sommet courant."""
            dist = {src: 0.0}
            pred_edge: dict[int, int] = {}
            pred_node: dict[int, int] = {}
            heap = [(0.0, src)]                # la file de priorite : (distance, sommet)
            while heap:
                # heappop retire et renvoie le plus petit element de la file. Les
                # couples sont compares sur leur premier terme, donc sur la distance.
                d, u = heapq.heappop(heap)
                # Une meme case peut avoir ete empilee plusieurs fois, avec des valeurs
                # differentes. Si celle qu'on vient de sortir n'est plus la meilleure
                # connue, c'est un reste perime : on l'ignore.
                if d > dist.get(u, math.inf):
                    continue
                for v, w, eidx in sadj[u]:     # chaque voisin de u
                    nd = d + w                 # ce que couterait d'atteindre v en passant par u
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        pred_edge[v] = eidx
                        pred_node[v] = u
                        heapq.heappush(heap, (nd, v))
            return dist, pred_edge, pred_node

        # Un Dijkstra par sommet impair, pour connaitre toutes les distances entre eux.
        shortest = {u: dijkstra(u) for u in odd}

        # Appariement glouton : a chaque tour, on relie les deux sommets impairs les
        # plus proches et on duplique les super-aretes du chemin qui les relie. On les
        # apparie TOUS, jusqu'au dernier : le graphe n'a alors plus aucun sommet
        # impair, donc il admet un circuit ferme, qui revient de lui-meme a son point
        # de depart. C'est ce qui evite d'avoir a refermer le trace apres coup.
        #
        # Cette strategie ne donne pas forcement l'appariement le moins couteux. La
        # solution exacte passe par l'algorithme des fleurs d'Edmonds, dont
        # l'implementation serait sans commune mesure avec le gain : mesure sur huit
        # dessins de test, le procede glouton trouve l'optimum sept fois sur huit.
        remaining = set(odd)
        while remaining:
            best_pair = None
            rem = list(remaining)
            for i in range(len(rem)):
                dist_i = shortest[rem[i]][0]        # le dictionnaire des distances depuis rem[i]
                for j in range(i + 1, len(rem)):
                    d = dist_i.get(rem[j], math.inf)
                    if best_pair is None or d < best_pair[0]:
                        best_pair = (d, rem[i], rem[j])
            _, u, v = best_pair
            _, pred_edge, pred_node = shortest[u]
            # On remonte le chemin de v jusqu'a u en suivant les predecesseurs, et on
            # ajoute une copie de chaque arete traversee. dict(...) fabrique une vraie
            # copie du dictionnaire : sans cela, on rangerait deux fois le meme objet.
            cur = v
            while cur != u:
                super_edges.append(dict(super_edges[pred_edge[cur]]))
                cur = pred_node[cur]
            remaining.discard(u)
            remaining.discard(v)

        # --- 4) Extraction du circuit, par l'algorithme de Hierholzer. ---
        # On reconstruit la liste des aretes touchant chaque sommet, car des aretes
        # dupliquees ont ete ajoutees a l'etape precedente.
        inc_s: dict[int, list[int]] = defaultdict(list)
        for idx, edge in enumerate(super_edges):
            inc_s[edge["a"]].append(idx)
            inc_s[edge["b"]].append(idx)

        used_s = [False] * len(super_edges)
        # pointer memorise, pour chaque sommet, ou l'on en etait dans sa liste d'aretes.
        # Sans lui, il faudrait rebalayer cette liste depuis le debut a chaque visite,
        # ce qui rendrait l'algorithme beaucoup plus lent.
        pointer: dict[int, int] = defaultdict(int)
        # Tous les degres etant pairs, le parcours est un circuit et son point de
        # depart est quelconque. Le test sur remaining n'est qu'un filet de securite.
        start = next(iter(remaining)) if remaining else super_edges[0]["a"]

        # LE PRINCIPE. On avance au hasard sans jamais reprendre une arete deja
        # utilisee. On ne peut se retrouver bloque qu'au point de depart : en arrivant
        # sur un autre sommet, on y a consomme un nombre impair d'aretes, or son degre
        # est pair, donc il en reste au moins une pour repartir. Quand on est bloque,
        # on revient en arriere en notant les aretes au passage ; les boucles laissees
        # de cote en chemin se recollent ainsi toutes seules.
        node_stack = [start]                       # la pile des sommets visites
        edge_stack: list[Optional[int]] = [None]   # l'arete empruntee pour atteindre chacun
        ordered_edges: list[int] = []
        while node_stack:
            v = node_stack[-1]                     # le sommet ou l'on se trouve
            # On avance le pointeur jusqu'a une arete encore libre.
            while pointer[v] < len(inc_s[v]) and used_s[inc_s[v][pointer[v]]]:
                pointer[v] += 1
            if pointer[v] < len(inc_s[v]):         # il en reste une : on l'emprunte
                ei = inc_s[v][pointer[v]]
                pointer[v] += 1
                used_s[ei] = True
                edge = super_edges[ei]
                nxt = edge["b"] if edge["a"] == v else edge["a"]
                node_stack.append(nxt)
                edge_stack.append(ei)
            else:                                  # plus rien : on revient en arriere
                node_stack.pop()
                ei = edge_stack.pop()
                if ei is not None:
                    ordered_edges.append(ei)
        # Comme les aretes sont enregistrees au moment du retour en arriere, elles
        # sortent dans l'ordre inverse du parcours.
        ordered_edges.reverse()

        # --- 5) Redeploiement en liste de points. ---
        result: list[tuple[float, float]] = [coords[start]]
        cur = start
        for ei in ordered_edges:
            edge = super_edges[ei]
            if edge["a"] == cur:             # on traverse la super-arete dans son sens
                pts = edge["pts"]
                nxt = edge["b"]
            else:                            # on la traverse a l'envers
                pts = edge["pts"][::-1]      # [::-1] renvoie la liste inversee
                nxt = edge["a"]
            # pts[1:] saute le premier point, deja present comme dernier point de
            # l'arete precedente, pour ne pas le repeter.
            result.extend(pts[1:])
            cur = nxt

        return result if len(result) >= 2 else None


# =========================
# Ecran dessin
# =========================

class DrawScreen(BaseScreen):
    """L'ecran ou l'utilisateur trace son dessin a la souris.

    Il y dispose d'un crayon, d'une gomme, de l'affichage des points, d'un effacement
    total, d'une fusion du dessin en un trace unique, et du passage a l'animation.

    Herite de BaseScreen : il dispose automatiquement du decoupage en zones, de la
    gestion des evenements et de l'affichage de l'interface commune, et ne redefinit
    que ce qui lui est propre."""

    caption = "Espace de Dessin"

    def __init__(self, app: "App"):
        """Prepare l'ecran de dessin et cree ses boutons.

        Arguments :
          app : l'application principale, transmise a la classe mere.

        Les boutons sont crees ici et non dans App parce qu'ils sont propres a cet
        ecran. Chacun recoit la methode qu'il devra declencher, passee sans
        parentheses pour que le bouton puisse l'appeler plus tard."""
        super().__init__(app)                # execute le constructeur de BaseScreen
        self.is_drawing = False              # True pendant qu'un trait est en cours de trace
        self.drawing_mode = False            # le crayon est-il actif ?
        self.erasing_mode = False            # la gomme est-elle active ?
        self.showing_points = False          # affiche-t-on un marqueur sur chaque point ?

        self.draw_button = Button("Dessiner", self.toggle_drawing, switch=True)
        self.erase_button = Button("Gommer", self.toggle_erasing, switch=True)
        self.points_button = Button("Afficher points", self.toggle_points, switch=True)
        self.animate_button = Button("Voir animation", self.go_to_animation)
        self.clear_button = Button("Tout effacer", self.clear)
        self.fill_button = Button("Boucher les trous", self.fill_gaps)
        self.buttons = [self.draw_button, self.erase_button, self.points_button,
                        self.animate_button, self.clear_button, self.fill_button]

    # --- Les commandes des boutons ---

    def toggle_drawing(self) -> None:
        """Active ou desactive le crayon. Appelee par le bouton Dessiner.

        Le crayon, la gomme et l'affichage des points s'excluent mutuellement : activer
        le crayon coupe donc les deux autres, et remet leurs boutons au repos."""
        self.drawing_mode = self.draw_button.active
        if self.drawing_mode:
            self.erasing_mode = False
            self.erase_button.active = False
            self.showing_points = False
            self.points_button.active = False

    def toggle_erasing(self) -> None:
        """Active ou desactive la gomme. Appelee par le bouton Gommer.

        Activer la gomme allume aussi l'affichage des points, pour que l'utilisateur
        voie ce qu'il efface, et coupe le crayon."""
        self.erasing_mode = self.erase_button.active
        if self.erasing_mode:
            self.drawing_mode = False
            self.draw_button.active = False
            self.showing_points = True
            self.points_button.active = True

    def toggle_points(self) -> None:
        """Allume ou eteint l'affichage d'un marqueur sur chaque point. Appelee par son bouton."""
        self.showing_points = self.points_button.active

    def go_to_animation(self) -> None:
        """Demande le passage a l'ecran d'animation. Appelee par le bouton Voir animation.

        Elle ne change pas d'ecran elle-meme : elle se contente de remplir
        next_screen, que la boucle principale lira a la prochaine image."""
        if len(self.state.paths) >= 1:
            self.next_screen = ScreenName.ANIMATE
        else:
            print("Animation impossible : il faut au moins un trace.")

    def clear(self) -> None:
        """Efface tout le dessin. Appelee par le bouton Tout effacer."""
        self.state.paths.clear()

    def fill_gaps(self) -> None:
        """Fusionne le dessin en un seul trace parcourable. Appelee par le bouton Boucher les trous.

        Confie le travail a GeometryEngine, puis remplace tout le dessin par le trace
        obtenu. L'utilisateur voit alors apparaitre les ponts, et peut encore les
        gommer ou les completer avant de lancer l'animation."""
        merged = GeometryEngine.build_single_path(self.state.paths)
        if merged is not None:
            # L'ecriture [:] remplace le contenu de la liste sans en creer une nouvelle,
            # ce qui est important car l'autre ecran garde une reference vers cette
            # meme liste.
            self.state.paths[:] = [merged]

    # --- Logique par image ---

    def update(self) -> None:
        """Lit la souris et trace ou gomme en consequence. Appelee une fois par image.

        Redefinit la methode vide de BaseScreen. Le programme ne reagit pas aux
        evenements de mouvement de la souris, mais interroge sa position a chaque
        image : c'est plus simple et suffisant a cette cadence."""
        # Le bouton d'animation est grise tant qu'il n'y a rien a animer.
        self.animate_button.locked = len(self.state.paths) == 0

        mouse_pos = pygame.mouse.get_pos()             # position de la souris, en pixels
        mouse_pressed = pygame.mouse.get_pressed()     # etat des trois boutons de la souris

        # On ignore tout ce qui se passe hors de la zone de dessin, ainsi que pendant
        # un deplacement de la bordure de la colonne.
        if not self.layout.rect_main.collidepoint(mouse_pos) or self.state.border_drag:
            self.is_drawing = False
            return

        # Conversion de la position en pixels vers les coordonnees logiques.
        rel_x = ((mouse_pos[0] - self.layout.rect_main.x) / self.layout.rect_main.width) * DRAWING_ZONE_RATIO
        rel_y = (mouse_pos[1] - self.layout.rect_main.y) / self.layout.rect_main.height

        if mouse_pressed[0] and self.drawing_mode:     # clic gauche maintenu, crayon actif
            self._add_drawing_point((rel_x, rel_y))
        elif mouse_pressed[0] and self.erasing_mode:   # clic gauche maintenu, gomme active
            self._erase_around((rel_x, rel_y))
        else:
            self.is_drawing = False                    # bouton relache : le trait en cours est fini

    def _add_drawing_point(self, point: tuple[float, float]) -> None:
        """Ajoute un point au trait en cours, en ouvrant un nouveau trait si necessaire.

        Arguments :
          point : la position de la souris en coordonnees logiques.

        Ne sert qu'a update. C'est ici que se joue le decoupage du dessin en traits :
        l'indicateur is_drawing passe a False des que l'utilisateur relache le bouton,
        si bien que le point suivant ouvrira automatiquement une nouvelle sous-liste."""
        if not self.is_drawing:
            self.is_drawing = True
            self.state.paths.append([])      # nouveau trait : une sous-liste vide de plus
        # On evite d'enregistrer deux fois le meme point, ce qui arriverait si la
        # souris ne bougeait pas d'une image a l'autre.
        if not self.state.paths[-1] or self.state.paths[-1][-1] != point:
            self.state.paths[-1].append(point)

    def _erase_around(self, point: tuple[float, float]) -> None:
        """Supprime les points proches du curseur et recoupe les traits en consequence.

        Arguments :
          point : la position de la souris en coordonnees logiques, centre de la gomme.

        Ne sert qu'a update. Gommer au milieu d'un trait le coupe en deux traits
        distincts : on reconstruit donc entierement la liste des traits, en ouvrant un
        nouveau morceau chaque fois qu'un point est supprime. Les morceaux d'un seul
        point sont abandonnes, un trait devant avoir au moins deux points pour exister."""
        radius = 0.03
        new_paths: list[list[tuple[float, float]]] = []
        for path in self.state.paths:
            current_segment: list[tuple[float, float]] = []
            for pt in path:
                # La division par le rapport de la zone corrige la difference d'echelle
                # entre les deux axes, l'abscisse allant jusqu'a 1.5 et l'ordonnee jusqu'a 1.
                distance = math.hypot((pt[0] - point[0]) / DRAWING_ZONE_RATIO, pt[1] - point[1])
                if distance > radius:
                    current_segment.append(pt)       # assez loin : on garde le point
                else:
                    if len(current_segment) > 1:     # on coupe : le morceau en cours est archive
                        new_paths.append(current_segment)
                    current_segment = []
            if len(current_segment) > 1:             # le dernier morceau du trait
                new_paths.append(current_segment)
        self.state.paths[:] = new_paths

    def draw(self) -> None:
        """Dessine l'interface commune, puis le dessin de l'utilisateur par-dessus.

        Redefinit la methode de BaseScreen, qu'elle commence par appeler."""
        self.draw_base_ui()
        for path in self.state.paths:
            if len(path) > 1:
                scaled_points = [self.to_screen(p) for p in path]
                # draw.lines relie une suite de points par des segments. Le False
                # signifie qu'on ne referme pas la figure, le 2 est l'epaisseur du trait.
                pygame.draw.lines(self.surface, (255, 255, 255), False, scaled_points, 2)
                if self.showing_points:
                    for pt in scaled_points:
                        pygame.draw.circle(self.surface, (255, 0, 0), pt, 2)
        pygame.display.flip()


# =========================
# Ecran animation
# =========================

class AnimateScreen(BaseScreen):
    """L'ecran ou l'animation se joue : les cercles tournants redessinent la courbe.

    A l'arrivee sur cet ecran, tout le calcul est effectue une fois : fusion du dessin,
    reechantillonnage, transformee de Fourier. Ensuite, chaque image se contente
    d'additionner les vecteurs tournants a l'instant courant et de les afficher."""

    caption = "Espace Animation"

    def __init__(self, app: "App"):
        """Prepare l'ecran d'animation, ses boutons et sa boite de reglage de vitesse.

        Arguments :
          app : l'application principale, transmise a la classe mere."""
        super().__init__(app)
        self.show_paths = False                      # superpose-t-on le dessin d'origine ?
        self.uni_path: Optional[np.ndarray] = None   # les points reechantillonnes
        self.coeffs: list[dict[str, float]] = []     # les cercles, calcules une fois pour toutes
        self.trace: list[tuple[float, float]] = []   # la courbe deja tracee par le crayon
        self.current_points: list[tuple[float, float]] = []  # bouts des vecteurs a l'instant courant
        self.t = 0.0                                 # le parametre de temps, dans [0 ; 1[
        self.loop_done = False                       # True une fois le premier tour acheve
        self.speed_value = SPEED_DEFAULT             # la valeur entiere reglee par la boite

        self.phantom_button = Button("Afficher le dessin", self.toggle_phantom, switch=True)
        self.back_button = Button("Retourner dessiner", self.back_to_draw)
        self.restart_button = Button("Revoir animation", self.restart)
        self.buttons = [self.phantom_button, self.back_button, self.restart_button]

        # L'unique boite de reglage du programme. Les deux lambda sont de petites
        # fonctions anonymes qui la relient a l'attribut speed_value : la premiere le
        # lit, la seconde l'ecrit. setattr(objet, "nom", valeur) affecte un attribut
        # designe par son nom sous forme de texte.
        self.value_boxes = [
            ValueBox("Vitesse",
                     getter=lambda: self.speed_value,
                     setter=lambda value: setattr(self, "speed_value", value),
                     min_val=SPEED_MIN, max_val=SPEED_MAX)
        ]

    @property
    def speed(self) -> float:
        """Renvoie la vitesse reellement utilisee dans les calculs.

        La boite de reglage manipule un entier de 1 a 1000, plus lisible a l'ecran ;
        les calculs ont besoin d'un nombre a virgule bien plus petit. @property permet
        d'ecrire simplement self.speed la ou on en a besoin, la conversion se faisant
        toute seule."""
        return self.speed_value / SPEED_DIVISOR

    def on_enter(self) -> None:
        """Prepare l'animation a chaque arrivee sur cet ecran.

        Redefinit la methode de BaseScreen en y ajoutant le calcul complet. C'est
        pourquoi un aller-retour vers l'ecran de dessin recalcule tout : si le dessin a
        change entre-temps, l'animation en tient compte."""
        super().on_enter()
        self.prepare_animation()
        self.phantom_button.active = False
        self.show_paths = False

    def prepare_animation(self) -> None:
        """Effectue tout le calcul prealable a l'animation.

        Enchaine trois etapes : obtenir un trace unique, le reechantillonner
        regulierement, puis en calculer les cercles. En cas de probleme a n'importe
        quelle etape, les coefficients sont vides et l'animation ne joue rien plutot
        que de planter.

        A noter : un dessin ne comportant qu'un seul trait est utilise directement,
        sans passer par le graphe, puisqu'il n'y a rien a relier ni a ordonner. C'est
        le seul cas ou le trace arrive ouvert, et c'est la fermeture de secours de
        FourierEngine qui s'en charge."""
        if len(self.state.paths) == 1:
            single_path = self.state.paths[0]
        else:
            single_path = GeometryEngine.build_single_path(self.state.paths)

        if single_path is None or len(single_path) < 2:
            self.uni_path = None
            self.coeffs = []
            return

        self.uni_path = FourierEngine.uniformize([single_path], NB_SAMPLES)
        if self.uni_path is None:
            self.coeffs = []
            return
        self.coeffs = FourierEngine.compute_dft(self.uni_path)
        self.restart()

    def progress(self) -> Optional[float]:
        """Indique l'avancement pour la barre de progression du bandeau.

        Redefinit la methode de BaseScreen, qui renvoie None par defaut. Ici, la
        reponse est le parametre de temps lui-meme, qui parcourt precisement [0 ; 1[
        pendant que le crayon trace la courbe entiere. Renvoie None tant qu'il n'y a
        rien a animer, ce qui fait disparaitre la barre."""
        if not self.coeffs or self.uni_path is None:
            return None
        return self.t

    def toggle_phantom(self) -> None:
        """Affiche ou masque le dessin d'origine en surimpression. Appelee par son bouton."""
        self.show_paths = self.phantom_button.active

    def back_to_draw(self) -> None:
        """Demande le retour a l'ecran de dessin. Appelee par son bouton."""
        self.next_screen = ScreenName.DRAW

    def restart(self) -> None:
        """Remet l'animation a son debut, sans refaire aucun calcul.

        Appelee par le bouton Revoir animation, et aussi a la fin de
        prepare_animation. Les coefficients sont conserves : seuls le temps et la
        courbe deja tracee sont remis a zero."""
        self.t = 0.0
        self.trace = []
        self.current_points = []
        self.loop_done = False

    def update(self) -> None:
        """Fait avancer le temps et recalcule la position de tous les vecteurs. Une fois par image.

        Redefinit la methode vide de BaseScreen. C'est ici qu'a lieu la reconstruction
        de Fourier proprement dite : on additionne les vecteurs tournants bout a bout,
        et le bout du dernier donne la position du crayon."""
        if not self.coeffs or self.uni_path is None:
            return
        # Le temps avance d'autant moins vite que les points sont nombreux, pour que
        # la duree d'un tour ne depende que du reglage de vitesse.
        self.t += self.speed / len(self.uni_path)
        if self.t > 1:
            self.t = 0
            self.loop_done = True                    # la courbe est entierement tracee

        # Somme des vecteurs tournants. current_points memorise chaque position
        # intermediaire, c'est-a-dire le bout de chaque vecteur, qui servira de centre
        # au cercle suivant lors de l'affichage.
        self.current_points = []
        curr_x, curr_y = 0.0, 0.0
        for coeff in self.coeffs:
            angle = 2 * np.pi * coeff["freq"] * self.t + coeff["phase"]
            curr_x += coeff["amp"] * math.cos(angle)     # partie reelle du nombre complexe
            curr_y += coeff["amp"] * math.sin(angle)     # partie imaginaire
            self.current_points.append((curr_x, curr_y))

        # On archive la position du crayon tant que le premier tour n'est pas fini ;
        # ensuite la courbe est complete et il n'y a plus rien a ajouter.
        if not self.loop_done:
            self.trace.append((curr_x, curr_y))

    def draw(self) -> None:
        """Dessine l'interface commune, puis les trois couches de l'animation.

        Redefinit la methode de BaseScreen. L'ordre compte : le dessin d'origine est
        au fond, les cercles au milieu, la courbe construite par-dessus."""
        self.draw_base_ui()
        if self.show_paths:
            self._draw_original_path()
        self._draw_epicycles()
        self._draw_trace()
        pygame.display.flip()

    def _draw_original_path(self) -> None:
        """Dessine le dessin d'origine en couleur tres sombre, en surimpression.

        Ne sert qu'a draw, et seulement quand l'option est activee. Permet de comparer
        a l'oeil la reconstruction et le trace initial."""
        for path in self.state.paths:
            if len(path) > 1:
                scaled = [self.to_screen(p) for p in path]
                pygame.draw.lines(self.surface, (60, 30, 0), False, scaled, 1)

    def _draw_epicycles(self) -> None:
        """Dessine la chaine de cercles tournants et le crayon a leur extremite.

        Ne sert qu'a draw. Le centre de chaque cercle est le bout du vecteur precedent :
        c'est precisement ce qui fait un epicycle, un cercle dont le centre est
        lui-meme en mouvement.

        Le premier coefficient, de frequence nulle, est la moyenne de tous les points
        du dessin, donc son centre de gravite. C'est un vecteur fixe qui ne tourne pas
        et sert seulement a placer le dessin au bon endroit : on ne dessine donc pas
        son cercle, d'ou la condition sur i."""
        if len(self.current_points) <= 1:
            return
        points_px = [self.to_screen(p) for p in self.current_points]
        for i, coeff in enumerate(self.coeffs):
            center = points_px[i - 1] if i > 0 else (self.layout.rect_main.x, self.layout.rect_main.y)
            end = points_px[i]
            radius = (coeff["amp"] / DRAWING_ZONE_RATIO) * self.layout.rect_main.width
            # On ignore les cercles de moins d'un demi-pixel, invisibles et nombreux.
            # Attention : ce filtre ne concerne QUE l'affichage. Ces coefficients sont
            # bel et bien additionnes par update et participent a la courbe.
            if radius > 0.5 and i > 0:
                pygame.draw.circle(self.surface, (50, 50, 50),
                                   (int(center[0]), int(center[1])), int(radius), 1)
                pygame.draw.circle(self.surface, (50, 20, 0),
                                   (int(end[0]), int(end[1])), 2)
        # Le point rouge au bout de la chaine : c'est lui le crayon.
        pygame.draw.circle(self.surface, (255, 0, 0),
                           (int(points_px[-1][0]), int(points_px[-1][1])), 4)

    def _draw_trace(self) -> None:
        """Relie les positions successives du crayon pour faire apparaitre la courbe.

        Ne sert qu'a draw. Le nombre de points de cette courbe depend du reglage de
        vitesse : a vitesse elevee, le crayon est echantillonne beaucoup moins souvent
        et le trace devient un polygone grossier."""
        if len(self.trace) > 1:
            trace_px = [self.to_screen(p) for p in self.trace]
            pygame.draw.lines(self.surface, (255, 255, 0), False, trace_px, 2)


# =========================
# Application principale
# =========================

class App:
    """L'objet racine du programme : il possede la fenetre et fait tourner la boucle.

    Il ignore completement ce que font les ecrans. Il se contente, a chaque image, de
    transmettre les evenements a l'ecran actif, de lui demander de se mettre a jour,
    puis de se dessiner. Cette ignorance est voulue : ajouter un troisieme ecran ne
    demanderait aucune modification ici."""

    def __init__(self):
        """Cree la fenetre, l'etat partage et les deux ecrans.

        Pygame ne gere qu'une seule fenetre, donc une seule surface de dessin. Les deux
        ecrans ne sont pas deux fenetres : ce sont deux objets qui dessinent tour a
        tour sur cette meme et unique surface, detenue ici."""
        # set_mode cree la fenetre et renvoie la surface sur laquelle tout sera dessine.
        # RESIZABLE autorise l'utilisateur a la redimensionner.
        self.surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()             # sert a limiter le nombre d'images par seconde
        self.font = pygame.font.SysFont("Arial", 20) # SysFont prend une police installee sur le systeme
        self.state = AppState()                      # l'etat partage, cree une seule fois

        # Les deux ecrans sont crees une fois pour toutes et ranges dans un
        # dictionnaire indexe par leur nom. Chacun recoit self, l'application, pour
        # pouvoir acceder a la fenetre et a l'etat partage.
        self.screens: dict[ScreenName, BaseScreen] = {
            ScreenName.DRAW: DrawScreen(self),
            ScreenName.ANIMATE: AnimateScreen(self),
        }
        self.current_screen_name = ScreenName.DRAW
        self.current_screen = self.screens[self.current_screen_name]
        self.current_screen.on_enter()

    def switch_to(self, screen_name: ScreenName) -> None:
        """Change d'ecran actif.

        Arguments :
          screen_name : le nom de l'ecran vers lequel basculer.

        Ne cree aucune fenetre : cela revient a designer quel objet recevra desormais
        les evenements et produira l'affichage. La surface est reaffectee au cas ou la
        fenetre aurait ete redimensionnee, puis on_enter reinitialise l'ecran
        d'arrivee."""
        self.current_screen_name = screen_name
        self.current_screen = self.screens[screen_name]
        self.current_screen.surface = self.surface
        self.current_screen.on_enter()

    def run(self) -> None:
        """Fait tourner la boucle principale jusqu'a la fermeture du programme.

        Chaque tour de boucle correspond a une image affichee et repete toujours les
        memes etapes : traiter les evenements accumules, appliquer un eventuel
        changement d'ecran, mettre a jour la logique, redessiner, attendre.

        Cette organisation est celle de toute application pygame."""
        running = True
        while running:
            # 1) Les evenements accumules depuis la derniere image. event.get vide la
            #    file d'attente et renvoie tout ce qui s'y trouvait.
            for event in pygame.event.get():
                self.current_screen.handle_event(event)

            # 2) L'ecran actif a-t-il demande un changement ? Il l'a signale en
            #    remplissant son attribut next_screen.
            if self.current_screen.next_screen == ScreenName.QUIT:
                running = False
            elif self.current_screen.next_screen in (ScreenName.DRAW, ScreenName.ANIMATE):
                self.switch_to(self.current_screen.next_screen)

            # 3) et 4) La logique, puis l'affichage.
            self.current_screen.update()
            self.current_screen.draw()

            # 5) tick attend le temps qu'il faut pour ne pas depasser la cadence
            #    voulue. Sans lui, la boucle tournerait aussi vite que la machine le
            #    permet, ce qui ferait chauffer le processeur pour rien.
            self.clock.tick(FPS)

        pygame.quit()    # ferme proprement tous les modules de pygame


# Ce test vaut True uniquement si le fichier est lance directement, et False s'il est
# importe depuis un autre fichier. Cela permet de reutiliser les classes ailleurs sans
# declencher l'ouverture d'une fenetre.
if __name__ == "__main__":
    App().run()
