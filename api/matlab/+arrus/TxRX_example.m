% Hej. Na prosbe PK pisze skrypt jak mniej wiecej wyobrazam sobie
% korzystanie z klas TxRx + komentarze. Skrypt ma w zamierzeniu charakter
% tymczasowy, bo byc moze jakies zmiany zajda, dlatego pozwalam sobie go pisac po polsku ;). 
%
% powiedzmy, ze chcemy zakodowac 3-krotne nadanie i odbior fali plaskiej przez
% aperture 192 elementowa.
%
% najpierw obiekt opisujacy impuls nadawczy. Zamknalem impuls nadawczy w
% klasie, bo kiedys moga byc arbitralne impulsy i wtedy moze sie to
% przydac. Na razie jednak obiekt jest b.prosty
pulse = TxPulse('nPeriods',[2], 'frequency', [5e6]);
% zastanawialem sie, czy by nie zrobic jakiegos domyslnego impulsu np.
% takiego jak wyzej, ale nie wiem, czy ze wzgledow bezpieczenstwa 
% nie lepiej przymusic uzytkownika do wygenerowania takiego obiektu - jak
% myslicie? W tej chwili domyslna wartoscia jest pusta tablica, ktora w
% zalozeniu (moim) miala oznaczac 'pusty strzal' czyli po prostu nic.

% potem generujemy obiekt opisujacy nadanie
t1 = Tx('pulse', pulse, 'aperture', true(1,192));

% nastepnie obiekt opisujacy odbior
r1 = Rx('aperture', true(1,192));

% nastepnie z tych dwoch obiektow produkujemy obiekt TxRX opisujacy
% pojedyncze nadanie/odbior.
txrx1 = TxRx('Tx',t1,'Rx', r1);
% Niestety, po napisaniu Tx i Rx zorientowalem sie, ze 
% tablice obiektow musza sie skladac z
% obiektow tego samego typu, wiec nie mozna bylo poprzestac tylko na
% obiektach Tx i Rx. Pierwotnie chcialem, zeby byla mozliwa sekwencja w
% stylu [tx1, tx2, tx3, rx1, tx4, tx5, tx6, rx2...], 
% dlatego potrzebny obiekt TxRx laczacy oba obiekty Tx i Rx.




% teraz budowana jest sekwencja (obiekt TxRxSequence) - na wejsciu podaje sie tablice obiektow
% TxRx, w tym wypadku 3 razy to samo tzn. [txrx1, txrx1, txrx1]
sequence = TxRxSequence([txrx1, txrx1, txrx1]);
% tutaj slowo komentarza - mozna pomyslec o tym, zeby w ogole nie bylo
% obiektu sequence (tak na poczatku chcialem zrobic), ale
% 1. jest tam miejsce na informacje o pri (sequence.pri), a nie wiem gdzie
%   by bylo logicznie ja umiescic (no, ewentualnie mozna by w samym kernelu, o
%   ktorym pozniej),
% 2. jesli bedzie taka potrzeba mozna tam trzymac metody do generowania
% sekwencji,
% 3. jest tam jakas kontrola, czy w tej liscie sa na pewno obiekty TxRX
% 4. mozna ewentualnie przerobic kernel, zeby lykal zarowno obiekty sequence jak i
% tablice TxRx.


% teraz o klasie TxRxKernel - wykorzystujac nasza sekwencje i obiekt
% sys (czyli ten, ktory jest uzywany w klasie Us4R) 
% mozna wewnatrz Us4R uzyc obiektu TxRxKernel do zaprogramowania sekwencji. 
% Zalozenie jest takie, ze w obiekcie Us4R bedzie gdzies jakas
% instrukcja warunkowa, ze jesli sekwencja jest obiektem TxRxSequence, to
% programowanie systemu bedzie przez kernel, ktory bedzie wygenerowany
% instrukcja w stylu:
kernel = TxRxKernel('sequence',sequence, 'usSystem', sys);
% samo programowanie jest zawarte w metodzie kernel.programHW() napisanej
% na postawie metody z Us4R o tej samej nazwie.


% a teraz uwagi:
% 1. Zalozylem, ze konstruktory wiekszosci tych obiektow, jesli sie nie
% poda odpowiednich parametrow na wejsciu, to odpowiednie propertisy sa
% puste. Musze w kernelu (chyba) dodac jeszcze jakies funkcje, ktore to
% badaja i w przypadku kiedy np. obiekt Rx ma pusta aperture, to wtedy
% kernel zamienia to na maske logiczna z samymi zerami (itd.)
%
% 2. Zalozylem, ze na razie (albo zawsze) bedzie to sluzyc do nadania i
% takze nie dotykalem rzeczy zwiazanych z rekonstrukcja. Miedzy innymi
% dlatego w kernelu nie ma ponizszej komendy obecnej w Us4R:
% Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
% Nie wiem czy to nie spowoduje jakis problemow (?). 
% Przy okazji pomyslalem, ze napisze wstepnie klase tgc, do opisu krzywej
% tgc, ale na razie ona jeszcze niczego nie robi, poza sprawdzeniem czy
% prawidlowe argumenty podane sa do konstruktora.
% 
% 3. Jeszcze tego nie testowalem. 


