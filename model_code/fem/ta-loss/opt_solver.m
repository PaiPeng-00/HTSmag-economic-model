function opt_solver()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'opt_solver.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
    m.param('par6').set('T','10[K]'); m.param('par6').set('Ip','583'); m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/4,td)');
    % --- 试1: 放宽 rtol + 放开初始步 (不动网格) ---
    fprintf('原 rtol=%s, initialstepbdf=%s\n', char(m.sol('sol1').feature('t1').getString('rtol')), char(m.sol('sol1').feature('t1').getString('initialstepbdf')));
    m.sol('sol1').feature('t1').set('tlist','range(0,td/4,td)');
    m.sol('sol1').feature('t1').set('rtol','1e-3');
    m.sol('sol1').feature('t1').set('initialstepbdfactive', false);
    fprintf('新 rtol=1e-3, initialstep=auto\n');
    fprintf('SOLVING... %s\n', datestr(now,31)); ts=tic; m.sol('sol1').runAll; el=toc(ts); fprintf('SOLVED %.0fs=%.1fmin (%s)\n', el, el/60, datestr(now,31));
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult;
    Dm=mphtable(m,'tmag').data;
    try; m.result.table.create('trad','Table'); catch; end
    m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult;
    Dr=mphtable(m,'trad').data;
    fprintf('=== 试1结果 (rtol=1e-3) ===\n');
    fprintf('磁化 int4: '); fprintf('%.3f ', Dm(:,2)); fprintf('-> peak=%.3f W (基准18.13)\n', max(Dm(:,2)));
    fprintf('径向 gev9: '); fprintf('%.1f ', Dr(:,2)); fprintf('-> peak=%.1f W (基准346.6)\n', max(Dr(:,2)));
    fprintf('用时 %.1f min (基准72min)\nOPT1_OK\n', el/60);
  catch ME
    fprintf(2,'OPT1_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
