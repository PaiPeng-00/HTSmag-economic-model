function build_arc_ta()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  try
    m=mphload([base 'TA_template_sparc.mph']); fprintf('LOADED\n');
    % KEEP SPARC turn/pancake structure (geometry arrays happy): Np=16, Nt=800, bands default
    % CHANGE only ARC dimensions + physics
    m.param.set('Rin','4.14[m]'); m.param.set('R2','0.40[m]'); m.param.set('wid','12[mm]');
    m.param('par6').set('Ip','660'); m.param('par6').set('T','20[K]');
    m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    Mrow=[1.47934e-05 5.83858e-06 2.8795e-06 1.61671e-06 9.90623e-07 6.55381e-07 4.68068e-07 3.63287e-07];
    for k=1:8; m.param('par5').set(sprintf('M1%d',k),sprintf('%.6g[H]',Mrow(k))); end
    m.func.create('Jcp','Analytic'); m.func('Jcp').set('funcname','Jcp');
    m.func('Jcp').set('expr','3268.2*b^(-0.6442)*exp(-(Tk-4.2)*log(1/(0.6781-0.0107*b))/17.8)');
    m.func('Jcp').set('args',{'Tk','b'}); m.func('Jcp').set('argunit',{'1','1'}); m.func('Jcp').set('fununit','1');
    m.component('comp1').variable('var1').set('Ic', ...
      'IcB20(abs(mf.Br))*Jcp(T[1/K],abs(mf.Br)[1/T]+1e-6)/Jcp(20,abs(mf.Br)[1/T]+1e-6)*wid/10[mm]*equ_turn');
    fprintf('rebuild geom (ARC dims, SPARC turn-structure)...\n'); m.geom('geom1').run; fprintf('GEOM_OK\n');
    fprintf('rebuild mesh...\n'); m.mesh('mesh1').run; fprintf('MESH_OK\n');
    mphsave(m,[base 'ARC_TA_loss.mph']); fprintf('SAVED\nBUILD_ARC_OK\n');
  catch ME
    fprintf(2,'BUILD_ERR_MSG: %s\n', ME.message);
  end
end
