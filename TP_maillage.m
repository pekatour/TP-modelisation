clear; close all;
% Nombre d'images utilisees
nb_images = 36; 

% chargement des images
for i = 1:nb_images
    if i<=10
        nom = sprintf('images/viff.00%d.ppm',i-1);
    else
        nom = sprintf('images/viff.0%d.ppm',i-1);
    end;
    % im est une matrice de dimension 4 qui contient 
    % l'ensemble des images couleur de taille : nb_lignes x nb_colonnes x nb_canaux 
    % im est donc de dimension nb_lignes x nb_colonnes x nb_canaux x nb_images
    im(:,:,:,i) = imread(nom); 
end;

% Affichage des images
figure; 
subplot(2,2,1); imshow(im(:,:,:,1)); title('Image 1');
subplot(2,2,2); imshow(im(:,:,:,9)); title('Image 9');
subplot(2,2,3); imshow(im(:,:,:,17)); title('Image 17');
subplot(2,2,4); imshow(im(:,:,:,25)); title('Image 25');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                                             %
% Calculs des superpixels                                 % 
% Conseil : afficher les germes + les régions             %
% à chaque étape / à chaque itération                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image = im(:,:,:,1);
N = size(image,1) * size(image,2);
k = 100;
S = round(sqrt(N/k));
max_iter = 1;
do_kmeans = 0;
m = 0.8 * S;

if do_kmeans
% initialiser centres
centers = zeros(floor(size(image,2)/S), floor(size(image,1)/S), 5);
centers(1,:,2) = floor(S/2);
centers(:,1,1) = floor(S/2);
for i=1:size(centers,1)
    for j=1:size(centers,2)
        if j ~= 1
            centers(i,j,1) = centers(i,j-1,1) + S;
        else
            if i ~= 1
                centers(i,j,2) = centers(i-1,j,2) + S;
            end
        end
        if i ~= 1
            centers(i,j,2) = centers(i-1,j,2) + S;
        else
            if j ~=1
                centers(i,j,1) = centers(i,j-1,1) + S;
            end
        end
    end
end
centers = reshape(centers(:),[],5);
for i=1:size(centers,1)
   centers(i,3:5) = image(centers(i,1),centers(i,2),:);
end

% affichage des centres initiaux
figure('Name',['Centres kmeans']);

% Affichage de la configuration initiale :
hold on;
imshow(image);
axis image;
axis off;
plot(centers(:,2),centers(:,1),'+','Color',"red",'LineWidth',1);
hold off;
pause(1);
% kmeans
[bestLabels, all_centers] = kmeans(image,centers,m,max_iter,S);

% affichage des itérations
for p=1:max_iter
    hold on;
    imshow(image);
    axis image;
    axis off;
    
    % affichage superpixels
    superpixels = [];
    for i=1:size(bestLabels,1)
        for j=1:size(bestLabels,2)
            current_value = bestLabels(i,j, p);
            if (i > 1 && bestLabels(i-1, j, p) ~= current_value) || ...  % Haut
               (i < size(centers,1) && bestLabels(i+1, j, p) ~= current_value) || ... % Bas
               (j > 1 && bestLabels(i, j-1, p) ~= current_value) || ...  % Gauche
               (j < size(centers,2) && bestLabels(i, j+1, p) ~= current_value)       % Droite
                superpixels = [superpixels; [i,j]];
            end
        end
    end
    plot(superpixels(:,2),superpixels(:,1),'.','Color',"green",'MarkerSize',0.1);
    title("Itération : ", sprintf('%d',p));

    % affichage germes
    plot(all_centers(:,2,p),all_centers(:,1,p),'+','Color',"red",'LineWidth',1);
    hold off;
    pause(1);
end


% ........................................................%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                                             %
% Binarisation de l'image à partir des superpixels        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ........................................................%

% Ici on a gardé les composantes rgb
last_bestLabels = bestLabels(:,:,max_iter);
last_centers_colors = all_centers(:,3:5,max_iter) ./ 255;

figure('Name',['Binarisation']);

Ysrgb = rgb2gray(last_centers_colors);
seuil = graythresh(Ysrgb);
classes = Ysrgb>seuil;
binarise = classes(last_bestLabels);
imshow(binarise);
title("Gray");
pause(1);

hsv = rgb2hsv(last_centers_colors);
hue = hsv(:,:,1);
seuil = graythresh(hue);
classes = hue>seuil;
binarise = classes(last_bestLabels);
imshow(binarise);
title("HSV");
pause(2);

hsv = rgb2lab(last_centers_colors);
hue = hsv(:,3);
seuil = graythresh(hue);
classes = hue>seuil;
binarise = classes(last_bestLabels);
hold on;
imshow(binarise);
title("LAB");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A FAIRE SI VOUS UTILISEZ LES MASQUES BINAIRES FOURNIS   %
% Chargement des masques binaires                         %
% de taille nb_lignes x nb_colonnes x nb_images           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ... 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET COMPLETER                              %
% quand vous aurez les images segmentées                  %
% Affichage des masques associes                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% déjà fait

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET COMPLETER                              %
% Frontière                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 1;
j = 1;
while i < size(binarise,1) && binarise(i,j) ~= 1
    for j=1:size(binarise,2)
        if binarise(i,j) == 1
            pfi = i;
            pfj = j;
            break;
        end
    end
    i = i + 1;
end

contour = bwtraceboundary(binarise,[pfi pfj],'W');
img_contour = zeros(size(binarise));
for i = 1:size(contour(:,1),1)
    img_contour(contour(i,1),contour(i,2)) = 1;
end

plot(contour(:,2),contour(:,1),'g','LineWidth',2);
pause(1);

contour_echant = contour(1:10:end,:);
[vx, vy] = voronoi(contour_echant(:,2), contour_echant(:,1));
vx_int = [];
vy_int = [];
for i=1:size(vx,2)
    if 0 < floor(vx(1,i)) && floor(vx(1,i)) < size(image,2) && 0 < floor(vy(1,i)) && floor(vy(1,i)) < size(image,1) && ...
            0 < floor(vx(2,i)) && floor(vx(2,i)) < size(image,2) && 0 < floor(vy(2,i)) && floor(vy(2,i)) < size(image,1)
        if binarise(floor(vy(1,i)), floor(vx(1,i))) > 0 && binarise(floor(vy(2,i)), floor(vx(2,i))) > 0
            vx_int= [vx_int [floor(vx(1,i)) ; floor(vx(2,i))]];
            vy_int= [vy_int [floor(vy(1,i)) ; floor(vy(2,i))]];
        end
    end

end
M = zeros(size(vx_int(:),1), 3);
M(:,1) = vx_int(:);
M(:,2) = vy_int(:);
D = bwdist(img_contour);
ind = sub2ind(size(D),M(:,2), M(:,1));
M(:,3) = D(ind);


plot(M(:,1),M(:,2),'.','color','m');
viscircles(M(:,1:2),M(:,3),'color','b', 'LineWidth', 0.5)
hold off;
pause(2);

% plot(vx_int,vy_int,'b','LineWidth',2);
hold off;
pause(2);

end

% chargement des points 2D suivis 
% pts de taille nb_points x (2 x nb_images)
% sur chaque ligne de pts 
% tous les appariements possibles pour un point 3D donne
% on affiche les coordonnees (xi,yi) de Pi dans les colonnes 2i-1 et 2i
% tout le reste vaut -1
pts = load('viff.xy');
% Chargement des matrices de projection
% Chaque P{i} contient la matrice de projection associee a l'image i 
% RAPPEL : P{i} est de taille 3 x 4
load dino_Ps;

% Reconstruction des points 3D
X = []; % Contient les coordonnees des points en 3D
color = []; % Contient la couleur associee
% Pour chaque couple de points apparies
for i = 1:size(pts,1)
    % Recuperation des ensembles de points apparies
    l = find(pts(i,1:2:end)~=-1);
    % Verification qu'il existe bien des points apparies dans cette image
    if size(l,2) > 1 & max(l)-min(l) > 1 & max(l)-min(l) < 36
        A = [];
        R = 0;
        G = 0;
        B = 0;
        % Pour chaque point recupere, calcul des coordonnees en 3D
        for j = l
            A = [A;P{j}(1,:)-pts(i,(j-1)*2+1)*P{j}(3,:);
            P{j}(2,:)-pts(i,(j-1)*2+2)*P{j}(3,:)];
            R = R + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),1,j));
            G = G + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),2,j));
            B = B + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),3,j));
        end;
        [U,S,V] = svd(A);
        X = [X V(:,end)/V(end,end)];
        color = [color [R/size(l,2);G/size(l,2);B/size(l,2)]];
    end;
end;
fprintf('Calcul des points 3D termine : %d points trouves. \n',size(X,2));

%affichage du nuage de points 3D
figure;
hold on;
for i = 1:size(X,2)
    plot3(X(1,i),X(2,i),X(3,i),'.','col',color(:,i)/255);
end;
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                  %
% Tetraedrisation de Delaunay  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = DelaunayTri(X(1,:)',X(2,:)',X(3,:)');                     

% A DECOMMENTER POUR AFFICHER LE MAILLAGE
fprintf('Tetraedrisation terminee : %d tetraedres trouves. \n',size(T,1));
% Affichage de la tetraedrisation de Delaunay
%figure;
%tetramesh(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcul des barycentres de chacun des tetraedres
poids = [1/2 1/6 1/6 1/6; 1/6 1/2 1/6 1/6; 1/6 1/6 1/2 1/6; 1/6 1/6 1/6 1/2];
%poids = [1/4 1/4 1/4 1/4];
nb_barycentres = size(poids,1);
C_g = zeros(4,size(T,1),nb_barycentres);
for i = 1:size(T,1)
    % Calcul des barycentres differents en fonction des poids differents
    for k = 1:nb_barycentres
        C_g(1:3,i,k)=sum(repmat(poids(k,:)',1,3) .* T.X(T.Triangulation(i,:),:),1);
        C_g(4,i,k) = 1;
    end
end

% A DECOMMENTER POUR VERIFICATION 
% A RE-COMMENTER UNE FOIS LA VERIFICATION FAITE
% Visualisation pour vérifier le bon calcul des barycentres*
%load mask;
% for i = 1:nb_images
%    for k = 1:nb_barycentres
%        o = P{i}*C_g(:,:,k);
%        o = o./repmat(o(3,:),3,1);
%        imshow(im_mask(:,:,i));
%        hold on;
%        plot(o(2,:),o(1,:),'rx');
%        pause;
%        close;
%    end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copie de la triangulation pour pouvoir supprimer des tetraedres
tri=T.Triangulation;
% Retrait des tetraedres dont au moins un des barycentres 
% ne se trouvent pas dans au moins un des masques des images de travail
% Pour chaque barycentre
load mask;

for t = size(tri,1):-1:1
    fini_t = 0;
    for i=1:size(im_mask,3)
        for k=1:nb_barycentres
            o = P{i}*C_g(:,t,k);
            o = o./o(3,1);
            if 0 < floor(o(1)) && floor(o(1)) < size(im_mask,1) && 0 < floor(o(2)) && floor(o(2)) < size(im_mask,2)
                if im_mask(floor(o(1)),floor(o(2)),i) == 0
                    tri(t,:) = [];
                    fini_t = 1;
                end
            end
            if fini_t
                break;
            end
        end
        if fini_t
            break;
        end
    end
end

% A DECOMMENTER POUR AFFICHER LE MAILLAGE RESULTAT
% Affichage des tetraedres restants
fprintf('Retrait des tetraedres exterieurs a la forme 3D termine : %d tetraedres restants. \n',size(tri,1));
figure;
trisurf(tri,X(1,:),X(2,:),X(3,:));

% Sauvegarde des donnees
save donnees;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSEIL : A METTRE DANS UN AUTRE SCRIPT %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load donnees;
% Calcul des faces du maillage à garder
% FACES = ...;
% ...

% fprintf('Calcul du maillage final termine : %d faces. \n',size(FACES,1));

% Affichage du maillage final
% figure;
% hold on
% for i = 1:size(FACES,1)
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,2))],[X(2,FACES(i,1)) X(2,FACES(i,2))],[X(3,FACES(i,1)) X(3,FACES(i,2))],'r');
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,3))],[X(2,FACES(i,1)) X(2,FACES(i,3))],[X(3,FACES(i,1)) X(3,FACES(i,3))],'r');
%    plot3([X(1,FACES(i,3)) X(1,FACES(i,2))],[X(2,FACES(i,3)) X(2,FACES(i,2))],[X(3,FACES(i,3)) X(3,FACES(i,2))],'r');
% end;
