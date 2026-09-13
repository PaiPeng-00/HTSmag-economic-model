function opt_mesh(numel_tape, hmax_tape)
  % 试2: rtol=1e-3 + 粗化带材网格 (dis1 每带材段数, size2 带材hmax)
  if nargin<1; numel_tape=15; end; if nargin<2; hmax_tape=0.006; end
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'opt_mesh.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
    m.param('par6').set('T','10[K]'); m.param('par6').set('Ip','583'); m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/4,td)');
    m.sol('sol1').feature('t1').set('tlist','range(0,td/4,td)');
    m.sol('sol1').feature('t1').set('rtol','1e-3'); m.sol('sol1').feature('t1').set('initialstepbdfactive', false);
    % 粗化带材网格
    m.mesh('mesh1').feature('dis1').set('numelem', numel_tape);   % 30 -> numel_tape
    m.mesh('mesh1').feature('size2').set('hmax', hmax_tape);       % 0.003 -> hmax_tape
    m.mesh('mesh1').run; ne=m.mesh('mesh1').getNumElem;
    fprintf('带材网格: dis1=%d段, size2 hmax=%g -> 单元=%d (原37022)\n', numel_tape, hmax_tape, ne);
    fprintf('SOLVING... %s\n', datestr(now,31)); ts=tic; m.sol('sol1').runAll; el=toc(ts); fprintf('SOLVED %.0fs=%.1fmin\n', el, el/60);
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
    try; m.result.table.create('trad','Table'); catch; end
    m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
    fprintf('=== 试2结果 (rtol=1e-3 + 网格粗化) ===\n');
    fprintf('磁化 peak=%.3f W (基准18.13), 径向 peak=%.1f W (基准346.6), 用时 %.1f min\nOPT2_OK\n', max(Dm(:,2)), max(Dr(:,2)), el/60);
  catch ME
    fprintf(2,'OPT2_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
