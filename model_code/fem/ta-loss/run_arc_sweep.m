function run_arc_sweep()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  if ~exist(res,'dir'); mkdir(res); end
  diary([res 'sweep.log']);
  fprintf('=== ARC sweep START %s ===\n', datestr(now,31));
  Npw_list=[5 10 20 100 200];
  Tlist=[4.2 10 20]; Ip_of=containers.Map([4.2 10 20],[973 844 660]);
  rho=5000;
  m=mphload([base 'ARC_TA_loss.mph']); fprintf('LOADED template\n');
  m.study('std1').feature('param').active(false);
  m.study('std1').feature('param1').active(false);
  m.study('std1').feature('time').set('tlist','range(0,td/24,td)');
  m.param('par7').set('rho_turn',[num2str(rho) '[uohm*cm^2]']);
  try; m.result.table.create('tmpmag','Table'); catch; end
  n=0; ntot=numel(Tlist)*numel(Npw_list);
  for T=Tlist
    for Npw=Npw_list
      n=n+1;
      csv=[res sprintf('ARC_magloss_Npw%d_rho%d_T%g.csv',Npw,rho,T)];
      if exist(csv,'file'); fprintf('[%d/%d] SKIP (done) Npw=%d T=%g\n',n,ntot,Npw,T); continue; end
      m.param('par6').set('T',[num2str(T) '[K]']);
      m.param('par6').set('Ip',num2str(Ip_of(T)));
      m.param('par2').set('Npw',num2str(Npw));
      fprintf('[%d/%d] SOLVING Npw=%d T=%g Ip=%d ... %s\n',n,ntot,Npw,T,Ip_of(T),datestr(now,31));
      ts=tic;
      try
        m.sol('sol1').runAll;
        m.result.numerical('int4').set('table','tmpmag');
        m.result.numerical('int4').set('data','dset1');
        m.result.numerical('int4').setResult;
        D=mphtable(m,'tmpmag').data;
        writematrix(D,csv);
        fprintf('   OK %.0fs  mag_peak=%.3f W/coil\n', toc(ts), max(D(:,2)));
      catch ME
        fprintf(2,'   FAIL Npw=%d T=%g: %s\n',Npw,T,ME.message);
      end
    end
  end
  fprintf('=== ARC sweep DONE %s ===\nSWEEP_COMPLETE\n', datestr(now,31));
  diary off;
end
