function cp5_solve_arc()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  if ~exist(res,'dir'); mkdir(res); end
  diary([res 'cp5_solve_arc.log']);
  try
    m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED %s\n', datestr(now,31));
    % single point: Npw=10, rho=5000, T=20K (already set in template)
    m.study('std1').feature('param').active(false);
    m.study('std1').feature('param1').active(false);
    % moderate time resolution over the 96h charge
    m.study('std1').feature('time').set('tlist','range(0,td/24,td)');
    fprintf('SOLVING single point (Npw=10,rho=5000,T=20K)... %s\n', datestr(now,31));
    ts=tic; m.sol('sol1').runAll; fprintf('SOLVED in %.0f s (%s)\n', toc(ts), datestr(now,31));
    % export mag-loss (int4) & radial-loss (gev3) vs time
    [t,mag]=safe2(@() mphglobal(m,{'t','int4'},'dataset','dset1','solnum','all'));
    [~,rad]=safe2(@() mphglobal(m,{'t','gev3'},'dataset','dset1','solnum','all'));
    writematrix([t(:) mag(:) rad(:)], [res 'ARC_loss_Npw10_rho5000_T20.csv']);
    fprintf('mag(end)=%g  radial(end)=%g\n', mag(end), rad(end));
    mphsave(m,[base 'ARC_TA_loss_solved.mph']);
    fprintf('SAVED solved mph + CSV\nCP5_SOLVE_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'CP5_SOLVE_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
function [a,b]=safe2(f); try; r=f(); a=r{1}; b=r{2}; catch; a=NaN; b=NaN; end; end
