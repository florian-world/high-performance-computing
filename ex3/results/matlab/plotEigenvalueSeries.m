

for b = [10 16 20 32 40 64]
% for b = 1
   

data = readmatrix(strcat("../data/facesOJA_eigenvalueseries_3_", int2str(b), ".csv"));
% data = readmatrix(strcat("../data/facesOJA_eigenvalueseries_test.csv"));

epoch = data(:,1);
lambdas = data(:,2:end);
[~, n] = size(lambdas);

true_lambdas = [4.390071632201753005e+02 2.895094656300519205e+02 1.360144553786574591e+02 9.800850485517051425e+01 6.322223392999567437e+01];
% true_lambdas = [3.45358035 0.4383435]

% [X0_1, X0_2] = 

true_lambdas_series = meshgrid(true_lambdas, ones(1,length(epoch)))';


figure2 = figure('Name', strcat("Sanger's Rule Eigenvalue Behavior n = ", int2str(n), ' b = ', int2str(b)));
% figure2 = figure('Name', strcat("Sanger's Rule Eigenvalue Behavior b = ", int2str(b)));
% figure2 = figure('Name', strcat("Sanger's Rule Faces Eigenvalue Behavior"));
axes2 = axes('Parent',figure2);
hold(axes2,'on');
title(figure2.Name);
xlabel("Epoch #");
ylabel("Eigenvalues");

colors = [...
         0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];

cmap = num2cell(colors(1:n,:), 2);

h = plot(epoch, lambdas,'LineWidth',2);
set(h, {'color'}, cmap);
h = plot(epoch, true_lambdas_series(1:n,:), 'LineWidth',1.25, 'LineStyle', '--');
set(h, {'color'}, cmap);
% xlim([1000 1800]);


% SAVE
set(gcf, 'PaperUnits', 'centimeters');
x_width=15 ;y_width=10;
set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
fileName = strcat('../figures/', ...
    regexprep(strrep(lower(figure2.Name), ' ', '-'), ...
              "[\[\]:=']", ''), '.eps');
saveas(gcf, fileName,'epsc');

 
end