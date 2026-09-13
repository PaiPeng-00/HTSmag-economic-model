function opt_maxstep(rtolv, maxstep_s, numel_tape)
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'opt_maxstep.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED\n');
    m.param('par6').set('T','10[K]'); m.param('par6').set('Ip','583'); m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/8,td)');
    t1=m.sol('sol1').feature('t1');
    t1.set('tlist','range(0,td/8,td)'); t1.set('rtol', num2str(rtolv)); t1.set('initialstepbdfactive', false);
    % 限制最大时间步
    try; t1.set('maxstepbdfactive', true); t1.set('maxstepbdf', num2str(maxstep_s)); fprintf('maxstep=%g s 已设\n', maxstep_s);
    catch e1; fprintf('maxstepbdf 属性名可能不对: %s\n', e1.message); end
    if numel_tape>0; m.mesh('mesh1').feature('dis1').set('numelem', numel_tape); m.mesh('mesh1').feature('size2').set('hmax', 0.005); m.mesh('mesh1').run; end
    fprintf('rtol=%g, maxstep=%gs, dis1=%d, 单元=%d\n', rtolv, maxstep_s, numel_tape, m.mesh('mesh1').getNumElem);
    fprintf('SOLVING... %s\n', datestr(now,31)); ts=tic; m.sol('sol1').runAll; el=toc(ts); fprintf('SOLVED %.0fs=%.1fmin\n', el, el/60);
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
    try; m.result.table.create('trad','Table'); catch; end
    m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
    fprintf('磁化 int4: '); fprintf('%.2f ', Dm(:,2)); fprintf('-> peak=%.2f (基准18.13, 应光滑)\n', max(Dm(:,2)));
    fprintf('径向 peak=%.1f (346.6), 用时 %.1f min\nMAXSTEP_OK\n', max(Dr(:,2)), el/60);
  catch ME
    fprintf(2,'MAXSTEP_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
